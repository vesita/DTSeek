"""DTSeek 多任务决策引擎演示：一次输入，三张任务卡同时出结果。

任务卡来源：checkpoints/multitask_v2_dtseek.pt
  1. pronoun   : 人称代词切片
  2. sentiment : 对话情绪切片
  3. ownership : 发言归属人切片

模型只输出 1-based 闭区间锚点 + 类别 + 置信度，不生成任何原文；
渲染层基于锚点索引直接在原句上做终端彩色高亮。
"""
import argparse
import os
from typing import Dict, List

import torch
import torch.nn.functional as F

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.nano_doc_encoder import NanoDocEncoder
from dtseek.robust_ar_model import RobustARSliceDecoder
from dtseek.segmenter import split_with_global_offsets

# 三任务各自的类别定义与配色（底色 + 亮字 + 下划线，行内对比度高）
TASK_STYLE = {
    "pronoun": [
        {"name": "无代词", "color": "\033[90m"},
        {"name": "第一人称", "color": "\033[1;96;44m"},   # 亮青 / 深蓝底
        {"name": "第二人称", "color": "\033[1;93;41m"},   # 亮黄 / 深红底
        {"name": "第三人称", "color": "\033[1;97;45m"},   # 亮白 / 品紫底
    ],
    "sentiment": [
        {"name": "中性", "color": "\033[90m"},
        {"name": "积极/喜悦", "color": "\033[1;92;40m"},  # 亮绿
        {"name": "愤怒/不满", "color": "\033[1;91;40m"},  # 亮红
        {"name": "悲伤/焦虑", "color": "\033[1;94;40m"},  # 亮蓝
    ],
    "ownership": [
        {"name": "无归属", "color": "\033[90m"},
        {"name": "用户/客户", "color": "\033[1;96;44m"},
        {"name": "助手/系统", "color": "\033[1;92;40m"},
        {"name": "第三方/团队", "color": "\033[1;95;45m"},
    ],
}
TASK_LABEL = {"pronoun": "人称代词", "sentiment": "情绪倾向", "ownership": "发言归属"}
RESET = "\033[0m"


class MultiTaskEngine:
    """共享 NanoDocEncoder 基座 + 多张可插拔任务卡。"""

    def __init__(self, ckpt_path: str = "checkpoints/multitask_v2_dtseek.pt", device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = NanoCharTokenizer()

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"未找到多任务权重 {ckpt_path}，请先跑 training/train_multitask.py")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        hidden_dim = ckpt["hidden_dim"]

        # 共享高效基座（nanoSeek 资产：RMSNorm + RoPE + QK-Norm + SwiGLU + FlashAttn）
        self.doc_encoder = NanoDocEncoder(
            vocab_size=self.tokenizer.vocab_size,
            hidden_dim=hidden_dim,
            num_layers=3,
            num_heads=4,
            max_len=128,
            dropout=0.0,
        ).to(self.device)
        self.doc_encoder.load_state_dict(ckpt["doc_encoder"])
        self.doc_encoder.eval()

        # 逐任务卡加载
        self.decoders: Dict[str, RobustARSliceDecoder] = {}
        for task, sd in ckpt["decoders"].items():
            dec = RobustARSliceDecoder(hidden_dim=hidden_dim, num_classes=4,
                                       num_heads=4, num_layers=2).to(self.device)
            dec.load_state_dict(sd)
            dec.eval()
            self.decoders[task] = dec

        self.tasks = list(self.decoders.keys())

    @torch.no_grad()
    def _run_segment(self, task: str, segment_text: str, max_steps: int = 4) -> List[Dict]:
        """在单个分句上跑一张任务卡的完整自回归发射。"""
        decoder = self.decoders[task]
        styles = TASK_STYLE[task]

        enc = self.tokenizer.encode(segment_text, max_length=64, padding=True)
        inp = torch.tensor([enc["input_ids"]], device=self.device)
        mask = torch.tensor([enc["attention_mask"]], dtype=torch.bool, device=self.device)
        L = len(segment_text)

        doc_memory = self.doc_encoder(inp, attention_mask=mask)
        q_seq = decoder.bos_query.clone()
        anchors, seen = [], set()

        for step in range(max_steps):
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
                "category": styles[pred_cls]["name"],
                "color": styles[pred_cls]["color"],
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

    def predict(self, text: str) -> Dict:
        """对所有任务卡并行输出（共享同一次基座编码，按需分句）。"""
        text = text.strip()
        if not text:
            return {"error": "输入为空"}

        segments = split_with_global_offsets(text, max_chunk_len=55)
        result = {"text": text, "num_segments": len(segments), "tasks": {}}

        for task in self.tasks:
            anchors = []
            for seg in segments:
                g0 = seg["global_start"]
                for a in self._run_segment(task, seg["text"]):
                    anchors.append({
                        "step": len(anchors) + 1,
                        "category": a["category"],
                        "color": a["color"],
                        "confidence": a["confidence"],
                        "s0": g0 + a["local_s0"],
                        "e0": g0 + a["local_e0"],
                        "next_action": a["next_action"],
                    })
            result["tasks"][task] = anchors

        return result


def render(text: str, anchors: List[Dict]) -> str:
    """按锚点索引在原文上做彩色下划线高亮。"""
    if not anchors:
        return text
    styles = [None] * len(text)
    for a in anchors:
        for i in range(a["s0"], a["e0"] + 1):
            if i < len(text):
                styles[i] = a["color"]
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


def show(predictor: MultiTaskEngine, line: str):
    res = predictor.predict(line)
    print("\n" + "=" * 68)
    print(f"输入: {res['text']}   (自动分句 {res['num_segments']} 段)")
    print("=" * 68)
    for task in res["tasks"]:
        anchors = res["tasks"][task]
        print(f"\n【{TASK_LABEL.get(task, task)}】{task}")
        print(f"  高亮: {render(res['text'], anchors)}")
        if not anchors:
            print("  切片: 无（背景/未触发）")
        else:
            for a in anchors:
                snip = res["text"][a["s0"]:a["e0"] + 1]
                print(f"  Step {a['step']}: {a['color']}[{a['s0']+1}:{a['e0']+1}]{RESET} "
                      f"{a['category']:10s} 置信度={a['confidence']:.3f} "
                      f"'{snip}' {a['next_action']}")
    print()


def main():
    ap = argparse.ArgumentParser(description="DTSeek 多任务决策引擎演示")
    ap.add_argument("text", nargs="?", help="待分析文本；留空进入交互模式")
    ap.add_argument("--ckpt", default="checkpoints/multitask_v2_dtseek.pt")
    args = ap.parse_args()

    engine = MultiTaskEngine(ckpt_path=args.ckpt)

    if args.text:
        show(engine, args.text)
        return

    print("\n" + "=" * 68)
    print("  DTSeek 多任务决策引擎（共享基座 + 可插拔任务卡）")
    print("  一次输入，同时给出：人称代词 / 情绪倾向 / 发言归属 三类切片")
    print("  输入 exit 退出")
    print("=" * 68)
    while True:
        try:
            line = input("\nDTSeek > ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                break
            show(engine, line)
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()
