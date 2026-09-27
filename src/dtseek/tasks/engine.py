"""多任务推理引擎 —— 基座常驻，任务卡**热插拔**。

插件化的意思是"换任务类型立刻换"，所以基座与任务卡是分开的两样东西：

    engine = MultiTaskEngine()                 # 加载基座一次（6.5 MB）
    engine.attach("cards/relation.pt")          # 挂一张卡（2.5 MB，秒级）
    engine.attach("cards/sentiment.pt")         # 再挂一张
    engine.predict(text)                        # 已挂的卡全跑
    engine.predict(text, tasks=["relation"])    # 只跑其中一张
    engine.attach("cards/relation_v2.pt")       # 同名 → 就地替换（不用重载基座）

加一张新卡不需要重训基座：用 `training/train_task_card.py` 在**冻结的基座**上单独训那张卡
（分钟级），存成一个小文件，attach 进来即可。

类别名、配色、窗口、切段策略都以**注册表里的任务卡**为准；产物里的 spec 快照只用来
校验漂移（见 `check_ckpt_specs`）。
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.encoder.segmenter import split_with_global_offsets
from dtseek.tasks.artifacts import (
    ArtifactError,
    build_card_decoder,
    check_card_base_compat,
    load_base_encoder,
    read_card,
)
from dtseek.tasks.plugin import TaskSpec, all_tasks, check_ckpt_specs
from dtseek.tasks.runtime import render_highlight

DEFAULT_BASE = "checkpoints/base_encoder.pt"
DEFAULT_CARDS_DIR = "checkpoints/cards"


class MultiTaskEngine:
    """基座常驻 + 可热插拔的任务卡。"""

    def __init__(self, base_path: str = DEFAULT_BASE, cards_dir: str | None = None,
                 device: str | None = None, auto_attach: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = NanoCharTokenizer()

        if not os.path.exists(base_path):
            raise FileNotFoundError(
                f"未找到基座产物 {base_path}。\n"
                f"  若手上是一体 ckpt，先拆：uv run python scripts/split_checkpoint.py --ckpt <ckpt>\n"
                f"  若要重训：uv run python training/train_multitask.py")

        try:
            self.doc_encoder, self._base = load_base_encoder(base_path, self.device)
        except ArtifactError as exc:
            raise ArtifactError(
                f"{exc}\n  拆分命令：uv run python scripts/split_checkpoint.py --ckpt {base_path}") from exc

        self.base_path = base_path
        self.specs: dict[str, TaskSpec] = {}
        self.decoders: dict[str, object] = {}
        self.card_paths: dict[str, str] = {}

        if auto_attach:
            self.attach_dir(cards_dir or DEFAULT_CARDS_DIR)

    # ---- 热插拔 -----------------------------------------------------------

    @property
    def attached(self) -> list[str]:
        return list(self.decoders)

    def attach(self, path: str | Path) -> str:
        """挂载（或**就地替换**同名的）一张任务卡。这是"立刻更换任务类型"的入口。"""
        path = str(path)
        ck = read_card(path)
        check_card_base_compat(ck, self._base, path)

        name = ck["task"]
        # 产物里的 spec 快照只用于校验漂移：类别名/类别数对不上就拦下来
        check_ckpt_specs({name: ck["spec"]}, all_tasks())

        decoder, artifact_spec = build_card_decoder(ck, self.device)
        registered = all_tasks().get(name)
        # 推理行为（切段策略/窗口）以注册表里的卡为准；卡没注册时退回产物里的快照
        self.specs[name] = registered.spec if registered is not None else artifact_spec
        self.decoders[name] = decoder
        self.card_paths[name] = path
        return name

    def attach_dir(self, directory: str | Path) -> list[str]:
        d = Path(directory)
        if not d.is_dir():
            return []
        return [self.attach(p) for p in sorted(d.glob("*.pt"))]

    def detach(self, name: str) -> None:
        self.decoders.pop(name, None)
        self.specs.pop(name, None)
        self.card_paths.pop(name, None)

    def task_label(self, task: str) -> str:
        return self.specs[task].label

    # ---- 推理 -------------------------------------------------------------

    @torch.no_grad()
    def _run_segment(self, task: str, segment_text: str) -> list[dict]:
        decoder = self.decoders[task]
        spec = self.specs[task]
        classes = spec.classes

        enc = self.tokenizer.encode(segment_text, max_length=spec.max_len, padding=True)
        inp = torch.tensor([enc["input_ids"]], device=self.device)
        mask = torch.tensor([enc["attention_mask"]], dtype=torch.bool, device=self.device)
        L = len(segment_text)

        doc_memory = self.doc_encoder(inp, attention_mask=mask)
        q_seq = decoder.bos_query.clone()
        anchors, seen = [], set()

        for step in range(spec.max_steps):
            out = decoder.forward_step(q_seq, doc_memory, doc_mask=mask)
            cls_prob = F.softmax(out["cls_logits"][0], dim=-1)
            pred_cls = int(cls_prob.argmax().item())
            action = int(F.softmax(out["action_logits"][0], dim=-1).argmax().item())

            if pred_cls == 0:
                break

            s_idx = int(out["start_logits"][0].argmax().item())
            e_idx = int(out["end_logits"][0].argmax().item())
            s0 = max(0, min(L - 1, min(s_idx, e_idx)))
            e0 = max(s0, min(L - 1, max(s_idx, e_idx)))

            if (s0, e0) in seen:
                break
            seen.add((s0, e0))

            anchors.append({
                "step": step + 1,
                "class_id": pred_cls,
                "category": classes[pred_cls].display,
                "color": classes[pred_cls].color,
                "confidence": round(float(cls_prob[pred_cls].item()), 4),
                "local_s0": s0,
                "local_e0": e0,
                "next_action": "<cont>" if action == 1 else "<eos>",
            })

            if action == 0:
                break

            next_q = decoder.get_step_input(
                prev_hidden=out["last_hidden"],
                prev_cls=torch.tensor([[pred_cls]], device=self.device),
                prev_start=torch.tensor([[[s0 / max(1, L)]]], device=self.device),
                prev_end=torch.tensor([[[e0 / max(1, L)]]], device=self.device),
            )
            q_seq = torch.cat([q_seq, next_q], dim=1)

        return anchors

    def predict(self, text: str, tasks: list[str] | None = None,
                max_chunk_len: int | None = None) -> dict:
        """对已挂载的（或指定的）任务卡输出。基座只编码一次，各卡共享。"""
        text = text.strip()
        if not text:
            return {"error": "输入为空"}

        chosen = self.attached if tasks is None else [t for t in tasks if t in self.decoders]
        result = {"text": text, "num_segments": 0, "tasks": {}}

        for task in chosen:
            spec = self.specs[task]
            limit = max_chunk_len or max(16, spec.max_len - 8)
            if spec.segment_policy == "window" and len(text) <= limit:
                # 整段一次解码：跨句状态（如人物 id）必须在窗口内保持一致，
                # 按句切开会让状态在句边界清零。训练也是整段喂的。
                segments = [{"text": text, "global_start": 0, "global_end": len(text)}]
            else:
                segments = split_with_global_offsets(text, max_chunk_len=limit)
            result["num_segments"] = max(result["num_segments"], len(segments))

            anchors = []
            for seg in segments:
                g0 = seg["global_start"]
                for a in self._run_segment(task, seg["text"]):
                    anchors.append({
                        **a,
                        "step": len(anchors) + 1,
                        "s0": g0 + a["local_s0"],
                        "e0": g0 + a["local_e0"],
                    })
            if spec.pair_emission:
                for i, a in enumerate(anchors):
                    a["pair_index"] = i // 2 + 1
                    a["pair_side"] = "左" if i % 2 == 0 else "右"
            result["tasks"][task] = anchors

        return result

    # ---- 呈现 -------------------------------------------------------------

    @staticmethod
    def render(text: str, anchors: list[dict]) -> str:
        return render_highlight(text, [(a["s0"], a["e0"], a["color"]) for a in anchors])
