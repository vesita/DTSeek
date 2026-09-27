"""任务卡的运行时 —— 与具体任务无关的编码、损失与验证。

这里只有三件事，全部由 `TaskSpec` 驱动：
  1. `GenericTaskDataset` : 把 `{text, spans}` 编成固定步数的自回归监督张量
  2. `task_loss`          : 单任务的多步教师强制损失
  3. `evaluate_task`      : 可判对错的逐任务指标

训练脚本、探针脚本、验收脚本都从这里取，避免各写一份实现后悄悄分叉。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dtseek.tasks.plugin import TaskSpec

RESET = "\033[0m"


class GenericTaskDataset(Dataset):
    """把 `{text, spans:[{label,start,end}]}` 编码成固定 `max_steps` 的监督张量。

    发射顺序 = span 按 start 升序；成对任务（`emission="pair"`）额外保证不会
    在两个成员之间截断，否则会出现「半个词对」这种无解监督。
    """

    def __init__(self, data: list[dict], tokenizer, spec: TaskSpec,
                 max_len: int | None = None, max_steps: int | None = None):
        self.data = data
        self.tokenizer = tokenizer
        self.spec = spec
        # 窗口按任务卡声明走：人物追踪要跨多轮上下文，窗口比别的卡大一倍
        self.max_len = spec.max_len if max_len is None else max_len
        self.max_steps = max_steps if max_steps is not None else spec.max_steps

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        enc = self.tokenizer.encode(item["text"], max_length=self.max_len, padding=True)
        spans = sorted(item["spans"], key=lambda x: x["start"])
        if self.spec.pair_emission:
            spans = spans[: len(spans) // 2 * 2]
        if self.spec.annotate_all and len(spans) > self.max_steps:
            # 静默截断 = 漏标一半的提及，却还训练模型"在这里收束"。宁可响亮地炸。
            raise ValueError(
                f"任务 {self.spec.name!r} 声明 annotate_all=True，但样本 idx={idx} 有 "
                f"{len(spans)} 个切片 > max_steps={self.max_steps}：{item['text']!r}。"
                " 数据集侧必须把样本卡在 max_steps 以内，不能靠运行期截断。")
        spans = spans[:self.max_steps]
        L = len(item["text"])

        labels = [0] * self.max_steps
        starts = [0] * self.max_steps
        ends = [0] * self.max_steps
        norm_starts = [0.0] * self.max_steps
        norm_ends = [0.0] * self.max_steps
        actions = [0] * self.max_steps
        step_mask = [0.0] * self.max_steps

        if len(spans) == 0:
            # 背景句：首步直接预测类别 0 + <eos>，之后所有步不计损失
            step_mask[0] = 1.0
        else:
            for i, s in enumerate(spans):
                step_mask[i] = 1.0
                labels[i] = s["label"]
                s_idx = min(self.max_len - 1, s["start"])
                e_idx = min(self.max_len - 1, max(s["start"], s["end"] - 1))
                starts[i] = s_idx
                ends[i] = e_idx
                norm_starts[i] = s_idx / max(1, L)
                norm_ends[i] = e_idx / max(1, L)
                actions[i] = 0 if (i == len(spans) - 1) else 1  # 0=<eos>, 1=<cont>

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long),
            "starts": torch.tensor(starts, dtype=torch.long),
            "ends": torch.tensor(ends, dtype=torch.long),
            "norm_starts": torch.tensor(norm_starts, dtype=torch.float),
            "norm_ends": torch.tensor(norm_ends, dtype=torch.float),
            "actions": torch.tensor(actions, dtype=torch.long),
            "step_mask": torch.tensor(step_mask, dtype=torch.float),
            "is_bg": torch.tensor(1.0 if len(spans) == 0 else 0.0, dtype=torch.float),
        }


def task_loss(decoder, doc_memory, mask, batch, spec: TaskSpec, device,
              ndb=None) -> torch.Tensor:
    """单任务的自回归多步损失（教师强制）。

    分类损失对背景类（id=0）加权：背景句在自回归范式下只在第 0 步贡献 **一个**
    监督信号，而有切片的句子贡献 N 个信号 —— 天然被双重稀释。轻微上调背景类权重
    可抑制"永远开火"的退化解（v1 的中性句 100% 误报）。

    `ndb`（`MentionNDB`，默认 None = 与旧行为逐位一致）：给身份槽任务接一段
    **情节检索记忆**。每一步先用（真值或指针给的）起始位置查表，把检索到的身份分布
    与分类头凸混合，再算损失；算出损失**之后**才把本步的真值 (起始字面 → label)
    写进记忆 —— 读写顺序保证不会把本步答案喂给自己（见 `mention_ndb.py`）。
    """
    max_steps = spec.max_steps
    t_labels = batch["labels"].to(device)
    t_starts = batch["starts"].to(device)
    t_ends = batch["ends"].to(device)
    t_nstarts = batch["norm_starts"].to(device)
    t_nends = batch["norm_ends"].to(device)
    t_actions = batch["actions"].to(device)
    step_mask = batch["step_mask"].to(device)
    B = doc_memory.shape[0]

    cls_w = torch.tensor(spec.cls_weights(), device=device)
    act_w = torch.tensor(list(spec.action_weight), device=device)

    input_ids = batch["input_ids"].to(device) if ndb is not None else None
    if ndb is not None:
        ndb.reset(B, device)          # 情节记忆：每批每条样本各一张空表

    q_seq = decoder.bos_query.expand(B, 1, -1)
    loss = torch.tensor(0.0, device=device)

    for s in range(max_steps):
        step_out = decoder.forward_step(q_seq, doc_memory, doc_mask=mask)
        cls_logits = step_out["cls_logits"]
        m = step_mask[:, s]
        if ndb is not None:
            # 读在写之前：此刻表里只有**更早**那些提及的绑定，本步答案还没写进去
            attn = ndb.read_attention(step_out["start_logits"], mask, t_starts[:, s])
            cls_logits = ndb.read(cls_logits, step_out["last_hidden"].squeeze(1),
                                  input_ids, attn)
        if m.sum() > 0:
            l_cls = (F.cross_entropy(cls_logits, t_labels[:, s],
                                     weight=cls_w, reduction="none") * m).sum() / m.sum()
            l_s = (F.cross_entropy(step_out["start_logits"], t_starts[:, s], reduction="none") * m).sum() / m.sum()
            l_e = (F.cross_entropy(step_out["end_logits"], t_ends[:, s], reduction="none") * m).sum() / m.sum()
            l_act = (F.cross_entropy(step_out["action_logits"], t_actions[:, s],
                                     weight=act_w, reduction="none") * m).sum() / m.sum()
            loss = loss + (l_cls + spec.span_weight * l_s + spec.span_weight * l_e + l_act)

        if ndb is not None:
            with ndb.write_enabled():
                ndb.write(step_out["last_hidden"].squeeze(1), input_ids,
                          t_starts[:, s], t_labels[:, s], m)

        # 教师强制：用真值切片状态驱动下一步（训练稳定、并行度高）
        next_q = decoder.get_step_input(
            prev_hidden=step_out["last_hidden"],
            prev_cls=t_labels[:, s:s + 1],
            prev_start=t_nstarts[:, s:s + 1, None],
            prev_end=t_nends[:, s:s + 1, None],
        )
        q_seq = torch.cat([q_seq, next_q], dim=1)

    return loss


@torch.no_grad()
def evaluate_task(doc_encoder, decoder, loader, device, spec: TaskSpec, ndb=None,
                  max_len: int = 64) -> dict:
    """逐任务可判对错验证。

    所有任务都测：
      - `cls_acc`  : 首切片类别正确率（分类对不对）
      - `span_hit` : 首切片起止区间与真值完全一致的比例（指针落点准不准）
      - `bg_fp`    : 背景句被误报出切片的比例（越低越好）

    以及**整句级**（首切片指标看不见"后面的切片漏了"）：
      - `exact_match`     : 整句切片集合与真值完全一致的比例
      - `slice_precision` / `slice_recall` : 切片级精确率与召回率

    `identity_labels=True` 的任务（人物1/2/3 这种匿名身份槽）另外测：
      - `first_mention_acc`  : 首次出现的提及 id 对不对（能靠出场顺序推）
      - `repeat_mention_acc` : **重复出现**的提及 id 对不对（这才是「id 有没有对上」）
      - `cluster_f1`         : 聚类一致性（成对口径，对 id 重命名不敏感）

    成对任务另外测：
      - `pair_exact` : 每一对「左右两个切片都命中且类别都对」的比例
      - `pair_order` : 每对内部是否严格左先右后（顺序错了对子就没用）
      - `pair_odd`   : 预测出的切片数是奇数的比例（隐式配对的天敌）
    """
    # 记住调用前的模式并原样还原：解码器 dropout=0.1，
    # 若在函数末尾硬写 .train()，调用方在它之后继续评估就会带 dropout，指标被污染。
    was_training = (doc_encoder.training, decoder.training)
    doc_encoder.eval()
    decoder.eval()

    cls_ok = cls_tot = 0
    span_ok = span_tot = 0
    bg_fired = bg_tot = 0
    pair_ok = pair_tot = 0
    order_ok = order_tot = 0
    odd_n = step_n = 0
    # 整句级：首切片指标会掩盖「后面的切片漏了」——实测代词卡首切片 100%
    # 而整句只有 95.8%，三个代词漏一个是完全看不见的。
    exact_n = exact_tot = 0
    tp = fp = fn = 0
    # 匿名身份槽任务：测「同一个人物的多次提及是否落在同一个 id 上」
    loc_n = loc_ok = 0
    first_n = first_ok = repeat_n = repeat_ok = 0
    pair_same_ok = pair_same_tot = pair_same_pred = 0

    for batch in loader:
        inp = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        doc_memory = doc_encoder(inp, attention_mask=mask)
        B = inp.shape[0]
        if ndb is not None:
            ndb.reset(B, device)      # 验证也是逐样本重建：绝不复用训练记忆

        # 首步：只判"第一个切片"——这是位置漂移与误报最直接的观测量
        q0 = decoder.bos_query.expand(B, 1, -1)
        out = decoder.forward_step(q0, doc_memory, doc_mask=mask)
        if ndb is not None:
            # 推理口径：读注意力只能来自**指针自己的预测**（没有真值可用）
            attn = ndb.read_attention(out["start_logits"], mask)
            out = dict(out, cls_logits=ndb.read(out["cls_logits"],
                                                out["last_hidden"].squeeze(1), inp, attn))

        pred_cls = out["cls_logits"].argmax(-1)          # [B]
        pred_s = out["start_logits"].argmax(-1)          # [B]
        pred_e = out["end_logits"].argmax(-1)            # [B]

        t_labels = batch["labels"][:, 0].to(device)
        t_s = batch["starts"][:, 0].to(device)
        t_e = batch["ends"][:, 0].to(device)
        is_bg = batch["is_bg"].to(device)

        # 1. 分类正确率（只看有切片的样本，背景句单列进 bg_fp）
        real = (t_labels > 0)
        cls_ok += ((pred_cls == t_labels) & real).sum().item()
        cls_tot += real.sum().item()

        # 2. 区间完全命中率（起止都对才算）
        span_ok += (((pred_s == t_s) & (pred_e == t_e)) & real).sum().item()
        span_tot += real.sum().item()

        # 3. 背景句误开火率（预测类别非 0 即为误报）
        bg_fired += ((pred_cls > 0) & (is_bg > 0.5)).sum().item()
        bg_tot += (is_bg > 0.5).sum().item()

        preds = _rollout(decoder, doc_memory, mask, B, spec, ndb=ndb, input_ids=inp)
        for b in range(B):
            truth = _truth_of(batch, b, spec)
            got = preds[b]
            # 整句完全一致（多重集比较：位置与类别都要对，数量也要对）
            exact_tot += 1
            if sorted(got) == sorted(truth):
                exact_n += 1
            ts, gs = set(truth), set(got)
            tp += len(ts & gs)
            fp += len(gs - ts)
            fn += len(ts - gs)
            if spec.identity_labels:
                pred_at = {(s0, e0): lab for lab, s0, e0 in got}
                seen_ids: set[int] = set()
                for lab, s0, e0 in truth:
                    if (s0, e0) not in pred_at:
                        seen_ids.add(lab)
                        continue
                    ok = pred_at[(s0, e0)] == lab
                    loc_n += 1
                    loc_ok += ok
                    if lab in seen_ids:
                        repeat_n += 1
                        repeat_ok += ok
                    else:
                        first_n += 1
                        first_ok += ok
                    seen_ids.add(lab)
                # 聚类一致性（成对口径，对 id 重命名不敏感）：
                # 真实同 id 的一对提及，预测是否也给了同一个 id；反之亦然
                spans = [(lab, s0, e0) for lab, s0, e0 in truth if (s0, e0) in pred_at]
                for i in range(len(spans)):
                    for j in range(i + 1, len(spans)):
                        same_true = spans[i][0] == spans[j][0]
                        same_pred = pred_at[(spans[i][1], spans[i][2])] == pred_at[(spans[j][1], spans[j][2])]
                        if same_true and same_pred:
                            pair_same_ok += 1
                        if same_true:
                            pair_same_tot += 1
                        if same_pred:
                            pair_same_pred += 1
            if spec.pair_emission and truth:
                pair_tot += 1
                step_n += 1
                if len(got) % 2 != 0:
                    odd_n += 1
                if _pairs_equal(got, truth):
                    pair_ok += 1
                order_tot += 1
                if _pair_order_ok(got):
                    order_ok += 1

    if was_training[0]:
        doc_encoder.train()
    if was_training[1]:
        decoder.train()
    report = {
        "cls_acc": cls_ok / max(1, cls_tot),
        "span_hit": span_ok / max(1, span_tot),
        "bg_fp": bg_fired / max(1, bg_tot),
        "n_cls": cls_tot,
        "n_bg": bg_tot,
        # 整句级（所有任务都有）：后段切片漏没漏，看这几个数
        "exact_match": exact_n / max(1, exact_tot),
        "slice_precision": tp / max(1, tp + fp),
        "slice_recall": tp / max(1, tp + fn),
    }
    if spec.identity_labels:
        # 重复提及那栏才是「人物 id 有没有对上」：首次提及本来就能靠出场顺序猜
        report.update({
            "id_acc": loc_ok / max(1, loc_n),
            "first_mention_acc": first_ok / max(1, first_n),
            "repeat_mention_acc": repeat_ok / max(1, repeat_n),
            "cluster_f1": (2 * pair_same_ok / max(1, pair_same_tot + pair_same_pred)),
            "n_measurable": loc_n,
        })
    if spec.pair_emission:
        report.update({
            "pair_exact": pair_ok / max(1, pair_tot),
            "pair_order": order_ok / max(1, order_tot),
            "pair_odd": odd_n / max(1, step_n),
            "n_pair": pair_tot,
        })
    return report


@torch.no_grad()
def _rollout(decoder, doc_memory, mask, B: int, spec: TaskSpec,
             ndb=None, input_ids=None) -> list[list[tuple[int, int, int]]]:
    """完整自回归发射一遍，返回每个样本的 [(label, start, end), ...]。

    接了 `ndb` 时，每步先按**预测的** start 指针查记忆，发射出切片后再把
    「预测的起始字面 → 预测的 label」写回该样本自己的情节记忆 —— 部署口径：
    记忆里存的全是模型自己发过的切片，没有任何真值。
    """
    q_seq = decoder.bos_query.expand(B, 1, -1)
    results: list[list[tuple[int, int, int]]] = [[] for _ in range(B)]
    alive = torch.ones(B, dtype=torch.bool, device=doc_memory.device)
    L = doc_memory.shape[1]
    if ndb is not None:
        ndb.reset(B, doc_memory.device)
    for _ in range(spec.max_steps):
        out = decoder.forward_step(q_seq, doc_memory, doc_mask=mask)
        if ndb is not None:
            attn = ndb.read_attention(out["start_logits"], mask)
            out = dict(out, cls_logits=ndb.read(out["cls_logits"],
                                                out["last_hidden"].squeeze(1), input_ids, attn))
        cls = out["cls_logits"].argmax(-1)
        s = out["start_logits"].argmax(-1)
        e = out["end_logits"].argmax(-1)
        act = out["action_logits"].argmax(-1)
        if ndb is not None:
            with ndb.write_enabled():
                ndb.write(out["last_hidden"].squeeze(1), input_ids, s, cls,
                          alive.to(cls.dtype))
        for b in range(B):
            if not alive[b]:
                continue
            if cls[b] == 0:
                alive[b] = False
                continue
            s0 = int(min(s[b], e[b]).clamp(0, L - 1))
            e0 = int(max(s[b], e[b]).clamp(0, L - 1))
            results[b].append((int(cls[b]), s0, e0))
            if act[b] == 0:
                alive[b] = False
        if not bool(alive.any()):
            break
        nxt = _step_input(decoder, out["last_hidden"], cls, s, e, L)
        q_seq = torch.cat([q_seq, nxt], dim=1)
    return results


def _step_input(decoder, last_hidden, cls, s, e, L: int):
    n = last_hidden.shape[0]
    dev = last_hidden.device
    ls = (s.float() / max(1, L)).clamp(0, 1).view(n, 1, 1)
    le = (e.float() / max(1, L)).clamp(0, 1).view(n, 1, 1)
    return decoder.get_step_input(
        prev_hidden=last_hidden,
        prev_cls=cls.view(n, 1).to(dev),
        prev_start=ls,
        prev_end=le,
    )


def _truth_of(batch: dict, b: int, spec: TaskSpec) -> list[tuple[int, int, int]]:
    out = []
    for i in range(spec.max_steps):
        if batch["step_mask"][b, i].item() < 0.5:
            break
        label = int(batch["labels"][b, i])
        if label == 0:
            break
        out.append((label, int(batch["starts"][b, i]), int(batch["ends"][b, i])))
    return out


def _pairs_equal(got: list, truth: list) -> bool:
    if len(got) != len(truth) or len(got) % 2 != 0:
        return False
    for i in range(0, len(got), 2):
        if got[i][0] != truth[i][0] or got[i][1] != truth[i][1] or got[i][2] != truth[i][2]:
            return False
        if got[i + 1][0] != truth[i + 1][0] or got[i + 1][1] != truth[i + 1][1] or got[i + 1][2] != truth[i + 1][2]:
            return False
    return True


def _pair_order_ok(got: list) -> bool:
    if len(got) % 2 != 0:
        return False
    return all(got[i][1] < got[i + 1][1] for i in range(0, len(got), 2))


def render_highlight(text: str, spans: list[tuple[int, int, str]]) -> str:
    """按 0-based 闭区间锚点在原文上打彩色下划线。后画的覆盖先画的。"""
    if not spans:
        return text
    styles: list[str | None] = [None] * len(text)
    for s0, e0, color in spans:
        for i in range(s0, e0 + 1):
            if 0 <= i < len(text):
                styles[i] = color
    out, cur = [], None
    for i, ch in enumerate(text):
        if styles[i] != cur:
            if cur is not None:
                out.append(RESET)
            if styles[i] is not None:
                out.append(f"\033[4m{styles[i]}")
            cur = styles[i]
        out.append(ch)
    if cur is not None:
        out.append(RESET)
    return "".join(out)
