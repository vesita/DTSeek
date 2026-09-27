"""多任务推理引擎 —— 共享基座 + 可插拔任务卡。

用法：
    engine = MultiTaskEngine("checkpoints/multitask_v2_dtseek.pt")
    engine.predict("我不太喜欢这个方案")
    engine.predict("...", tasks=["sentiment"])      # 只跑其中一张卡

类别名、展示名、配色、发射步数全部来自 ckpt 里存的 `TaskSpec` 快照 ——
推理端不再手抄第二份类别定义。加载时用 `check_ckpt_specs` 做 fail-closed 校验：
ckpt 与当前代码的任务声明对不上就直接抛错，而不是静默跑出错的结果。
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.encoder.nano_doc_encoder import NanoDocEncoder
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder
from dtseek.encoder.segmenter import split_with_global_offsets
from dtseek.tasks.plugin import TaskSpec, all_tasks, check_ckpt_specs
from dtseek.tasks.runtime import render_highlight

DEFAULT_CKPT = "checkpoints/multitask_frozen_dtseek.pt"


class MultiTaskEngine:
    """加载一个多任务 ckpt，按任务卡逐张推理。"""

    def __init__(self, ckpt_path: str = DEFAULT_CKPT, device: str | None = None,
                 verify_specs: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = NanoCharTokenizer()

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"未找到多任务权重 {ckpt_path}，请先跑 training/train_multitask.py")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "task_specs" not in ckpt:
            raise ValueError(
                f"{ckpt_path} 是插件化之前的旧格式（只有类别名、没有 TaskSpec 快照）。"
                " 请用当前的 training/train_multitask.py 重新训练。")

        self.specs: dict[str, TaskSpec] = {
            name: TaskSpec.from_snapshot(snap) for name, snap in ckpt["task_specs"].items()
        }
        if verify_specs:
            for note in check_ckpt_specs(ckpt["task_specs"], all_tasks()):
                print(f"  [ckpt] {note}")

        # 推理行为以**注册表里的任务卡**为准：快照记录的是"权重当年照什么训的"，
        # 而切段策略、窗口大小这类推理行为属于代码。快照里写了且不一致的，上面已拦掉；
        # 快照里没写的（旧格式），就在这里按代码继承。
        for name, card in all_tasks().items():
            if name in self.specs:
                self.specs[name] = card.spec

        hidden_dim = ckpt["hidden_dim"]
        enc_kwargs = dict(ckpt.get("encoder_kwargs", {}))
        dec_kwargs = dict(ckpt.get("decoder_kwargs", {"num_heads": 4, "num_layers": 2}))

        self.doc_encoder = NanoDocEncoder(
            vocab_size=self.tokenizer.vocab_size,
            hidden_dim=hidden_dim,
            dropout=0.0,
            **enc_kwargs,
        ).to(self.device)
        self.doc_encoder.load_state_dict(ckpt["doc_encoder"])
        self.doc_encoder.eval()

        self.decoders: dict[str, RobustARSliceDecoder] = {}
        for name, sd in ckpt["decoders"].items():
            dec = RobustARSliceDecoder(hidden_dim=hidden_dim,
                                       num_classes=self.specs[name].num_classes,
                                       **dec_kwargs).to(self.device)
            dec.load_state_dict(sd)
            dec.eval()
            self.decoders[name] = dec

        self.tasks: list[str] = list(ckpt.get("task_order") or sorted(self.decoders))
        self.train_args = ckpt.get("train_args", {})

    # ---- 推理 -------------------------------------------------------------

    @torch.no_grad()
    def _run_segment(self, task: str, segment_text: str) -> list[dict]:
        """在单个分句上跑一张任务卡的完整自回归发射。"""
        decoder = self.decoders[task]
        spec = self.specs[task]
        classes = spec.classes

        enc = self.tokenizer.encode(segment_text, max_length=64, padding=True)
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
        """对所有（或指定）任务卡并行输出，按需分句。

        **切段长度必须逐任务取**：人物追踪声明 `max_len=120`（要跨多轮上下文），
        若这里仍用统一的 55 字去切，一段人物对话会被劈成几段、id 在每段里从 1 重来，
        跨段共指直接失效。实测这就是「重复提及拿到错误 id + 62% 漏标」的原因。
        """
        text = text.strip()
        if not text:
            return {"error": "输入为空"}

        chosen = self.tasks if tasks is None else [t for t in tasks if t in self.decoders]
        result = {"text": text, "num_segments": 0, "tasks": {}}

        for task in chosen:
            spec = self.specs[task]
            limit = max_chunk_len or max(16, spec.max_len - 8)
            if spec.segment_policy == "window" and len(text) <= limit:
                # 整段一次解码：跨句共指要求 id 在窗口内保持一致，
                # 按句切开会让 id 每句从 1 重来（而且训练就是这么整段喂的）。
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
            if self.specs[task].pair_emission:
                for i, a in enumerate(anchors):
                    a["pair_index"] = i // 2 + 1
                    a["pair_side"] = "左" if i % 2 == 0 else "右"
            result["tasks"][task] = anchors

        return result

    def task_label(self, task: str) -> str:
        return self.specs[task].label

    # ---- 呈现 -------------------------------------------------------------

    @staticmethod
    def render(text: str, anchors: list[dict]) -> str:
        return render_highlight(text, [(a["s0"], a["e0"], a["color"]) for a in anchors])
