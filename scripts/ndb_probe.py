#!/usr/bin/env python
"""Mention-NDB 的**配对机制探针**：把「检索有没有用」与「自由 rollout 丢没丢」分开量。

`evaluate_task` 的数字来自**自由 rollout**（模型用自己预测的片段往前跑），所以它把三件事
混在一起：(a) 检索对不对，(b) 门控开没开，(c) rollout 误差累积。本探针只改一件事：
把自回归调度固定成**教师强制**（每步喂真值片段），然后在**同一批数据、同一条前向路径**上
跑三个 pass：

    head_only     : 读门控强行归零  → 纯分类头（= 不接 NDB 的等价物）
    ndb_true      : 正常，读位置用**真值起点**  → 训练时的口径
    ndb_pred      : 正常，读位置用**指针 argmax**（硬） → 推理口径
    ndb_pred_soft : 正常，读位置用**指针 softmax**（软） → 已被证伪的反例

三个 pass 的输入逐位相同，所以 Δ 是配对的。另外报「纯检索 top-1 命中率」（argmax p_ng
是否等于真值）——这是判断「记忆里到底有没有那个答案」的唯一干净口径。

    uv run python scripts/ndb_probe.py --ckpt /tmp/ab_ndb/person_ndb_seed42.pt --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.decoder.mention_ndb import MentionNDB  # noqa: E402
from dtseek.tasks.artifacts import build_card_decoder, load_base_encoder, read_card  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset  # noqa: E402

SAMPLES = {"pronoun": 6000, "sentiment": 32000, "relation": 14000,
           "person": 9000, "idiom": 14000, "ownership": 8000}


@torch.no_grad()
def _pass(name, ndb, decoder, doc_encoder, loader, device, spec, mode, batch_size):
    """一个 pass 的教师强制前向，返回 first/repeat/id 三项与检索诊断。"""
    head_bias_backup = None
    if mode == "head_only" and ndb is not None:
        head_bias_backup = ndb.read_gate.bias.detach().clone()
        ndb.read_gate.bias.fill_(-1e9)          # g ≡ 0：与「不接 NDB」逐位等价

    first_n = first_ok = rep_n = rep_ok = 0
    for batch in loader:
        inp = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        mem = doc_encoder(inp, attention_mask=mask)
        t_labels = batch["labels"].to(device)
        t_starts = batch["starts"].to(device)
        step_mask = batch["step_mask"].to(device)
        t_nstarts = batch["norm_starts"].to(device)
        t_nends = batch["norm_ends"].to(device)
        B = inp.shape[0]
        if ndb is not None:
            ndb.reset(B, device)
        seen = [set() for _ in range(B)]
        q_seq = decoder.bos_query.expand(B, 1, -1)
        for s in range(spec.max_steps):
            out = decoder.forward_step(q_seq, mem, doc_mask=mask)
            cls_logits = out["cls_logits"]
            if ndb is not None:
                true_starts = t_starts[:, s] if mode == "ndb_true" else None
                attn = ndb.read_attention(out["start_logits"], mask, true_starts,
                                          hard=(mode != "ndb_pred_soft"))
                tgt = torch.where(step_mask[:, s] > 0.5, t_labels[:, s],
                                  torch.full_like(t_labels[:, s], -100))
                cls_logits = ndb.read(cls_logits, out["last_hidden"].squeeze(1), inp, attn,
                                      targets=tgt)
            pred = cls_logits.argmax(-1)
            for b in range(B):
                if step_mask[b, s].item() < 0.5:
                    continue
                lab = int(t_labels[b, s])
                if lab == 0:
                    continue
                ok = int(pred[b]) == lab
                if lab in seen[b]:
                    rep_n += 1
                    rep_ok += int(ok)
                else:
                    first_n += 1
                    first_ok += int(ok)
                    seen[b].add(lab)
            if ndb is not None:
                with ndb.write_enabled():
                    ndb.write(out["last_hidden"].squeeze(1), inp, t_starts[:, s], t_labels[:, s],
                              step_mask[:, s])
            nxt = decoder.get_step_input(
                prev_hidden=out["last_hidden"], prev_cls=t_labels[:, s:s + 1],
                prev_start=t_nstarts[:, s:s + 1, None], prev_end=t_nends[:, s:s + 1, None])
            q_seq = torch.cat([q_seq, nxt], dim=1)

    if head_bias_backup is not None:
        with torch.no_grad():
            ndb.read_gate.bias.copy_(head_bias_backup)

    res = {
        "pass": name,
        "first_mention_acc": first_ok / max(1, first_n),
        "repeat_mention_acc": rep_ok / max(1, rep_n),
        "id_acc": (first_ok + rep_ok) / max(1, first_n + rep_n),
        "n_first": first_n, "n_repeat": rep_n,
    }
    if ndb is not None:
        st = ndb.stats()
        res.update({k: st[k] for k in ("retrieval_top1_hit", "retrieval_covered", "n_retrieval",
                                       "last_gate", "last_covered", "alpha")})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mention-NDB 配对机制探针（教师强制）")
    ap.add_argument("--ckpt", required=True, help="带 NDB 的任务卡产物")
    ap.add_argument("--card", default="person")
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--seed", type=int, default=42, help="必须与训练时相同（决定 val 切分）")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    card = resolve_tasks([args.card])[args.card]
    spec = card.spec
    ck = read_card(args.ckpt)
    ndb_extra = ck.get("extra", {}).get("ndb")
    if not ndb_extra:
        raise SystemExit(f"{args.ckpt} 里没有 NDB 状态（extra.ndb），不是接 NDB 训出来的卡")

    doc_encoder, _ = load_base_encoder(args.base, device)
    decoder, _ = build_card_decoder(ck, device)
    ndb = MentionNDB(**ndb_extra["kwargs"]).to(device)
    ndb.load_state_dict(ndb_extra["state_dict"])
    ndb.eval()
    print(f"加载 {args.ckpt}：NDB levels={ndb.levels} slots={ndb.slots} "
          f"read_true={ndb.read_true} | 表(batch={args.batch_size}) {ndb.table_gb(args.batch_size):.4f}GB")

    samples = args.samples or SAMPLES.get(args.card, 8000)
    data = card.build_dataset(samples)
    random.Random(args.seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    val = data[:n_val]
    loader = DataLoader(GenericTaskDataset(val, NanoCharTokenizer(), spec),
                        batch_size=args.batch_size, shuffle=False)

    runs = []
    for name, mode in (("head_only", "head_only"), ("ndb_true", "ndb_true"),
                       ("ndb_pred", "ndb_pred"), ("ndb_pred_soft", "ndb_pred_soft")):
        ndb.reset_stats()
        runs.append(_pass(name, ndb, decoder, doc_encoder, loader, device, spec, mode, args.batch_size))

    print("\n============ 教师强制下的身份决策（同一批 val，配对）============")
    for r in runs:
        extra = ""
        if "retrieval_top1_hit" in r:
            extra = (f"  纯检索hit={r['retrieval_top1_hit']:.4f}"
                     f" 覆盖={r['retrieval_covered']:.4f} gate={r['last_gate']:.3f}")
        print(f"{r['pass']:>10}: first={r['first_mention_acc']:.4f} "
              f"repeat={r['repeat_mention_acc']:.4f} id={r['id_acc']:.4f} "
              f"(n_first={r['n_first']} n_repeat={r['n_repeat']}){extra}")
    head = runs[0]
    for r in runs[1:]:
        print(f"Δ({r['pass']} − head_only): "
              f"first={r['first_mention_acc'] - head['first_mention_acc']:+.4f}  "
              f"repeat={r['repeat_mention_acc'] - head['repeat_mention_acc']:+.4f}  "
              f"id={r['id_acc'] - head['id_acc']:+.4f}")
    print("\nNDB_PROBE " + json.dumps(runs, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
