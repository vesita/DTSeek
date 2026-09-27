"""DTSeek 对话情绪切片分类演示（情绪专卡）。

重要：本脚本现在加载 `checkpoints/multitask_v2_dtseek.pt` 里的 **sentiment 任务卡**，
而不再是旧的 `sentiment_adapter_dtseek.pt`。
旧 adapter 用的是被淘汰的 SimpleDocEncoder + 61 词词典，会把"难受"判成积极；
新权重来自 NanoDocEncoder 高效基座 + 187 词扩充词典 + 三任务联合训练。

情绪体系：
  - 0: 中性 (灰)
  - 1: 积极 / 喜悦 / 赞赏 (亮绿黑底)
  - 2: 愤怒 / 暴躁 / 不满 (亮红黑底)
  - 3: 悲伤 / 沮丧 / 焦虑 (亮蓝黑底)

模型只输出 1-based 闭区间锚点，不生成原文；渲染层据此在原文上高亮。
支持任意长文本：先由 segmenter 分句，再把局部锚点映射回全局字符坐标。
"""
import argparse
import os

import torch
import torch.nn.functional as F

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.nano_doc_encoder import NanoDocEncoder
from dtseek.robust_ar_model import RobustARSliceDecoder
from dtseek.segmenter import split_with_global_offsets

CLASSES = [
    {"id": 0, "name": "中性/客观", "color": "\033[90m"},
    {"id": 1, "name": "积极/喜悦", "color": "\033[1;92;40m"},
    {"id": 2, "name": "愤怒/不满", "color": "\033[1;91;40m"},
    {"id": 3, "name": "悲伤/焦虑", "color": "\033[1;94;40m"},
]
RESET = "\033[0m"

DEFAULT_CKPT = "checkpoints/multitask_v2_dtseek.pt"


class SentimentSlicePredictor:
    def __init__(self, ckpt_path: str = DEFAULT_CKPT, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = NanoCharTokenizer()

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"未找到 {ckpt_path}。请先运行 `uv run python training/train_multitask.py` 生成。")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "decoders" not in ckpt or "sentiment" not in ckpt["decoders"]:
            raise ValueError(
                f"{ckpt_path} 里没有 sentiment 任务卡。\n"
                f"注意：旧的 checkpoints/sentiment_adapter_dtseek.pt 已被淘汰"
                f"（61 词词典 + 旧编码器，会把'难受'判成积极），请改用多任务权重。")

        hidden_dim = ckpt["hidden_dim"]

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

        self.decoder = RobustARSliceDecoder(
            hidden_dim=hidden_dim, num_classes=4, num_heads=4, num_layers=2
        ).to(self.device)
        self.decoder.load_state_dict(ckpt["decoders"]["sentiment"])
        self.decoder.eval()

    @torch.no_grad()
    def _run_segment(self, segment_text: str, max_steps: int = 4) -> list[dict]:
        enc = self.tokenizer.encode(segment_text, max_length=64, padding=True)
        inp = torch.tensor([enc["input_ids"]], device=self.device)
        mask = torch.tensor([enc["attention_mask"]], dtype=torch.bool, device=self.device)
        L = len(segment_text)

        doc_memory = self.doc_encoder(inp, attention_mask=mask)
        q_seq = self.decoder.bos_query.clone()
        anchors, seen = [], set()

        for step in range(max_steps):
            out = self.decoder.forward_step(q_seq, doc_memory, doc_mask=mask)
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
                "category": CLASSES[pred_cls]["name"],
                "color": CLASSES[pred_cls]["color"],
                "confidence": round(float(cls_prob[pred_cls].item()), 4),
                "local_s0": s0,
                "local_e0": e0,
                "next_action": "<cont>" if action == 1 else "<eos>",
            })
            if action == 0:
                break

            next_q = self.decoder.get_step_input(
                prev_hidden=out["last_hidden"],
                prev_cls=torch.tensor([[pred_cls]], device=self.device),
                prev_start=torch.tensor([[[s0 / max(1, L)]]], device=self.device),
                prev_end=torch.tensor([[[e0 / max(1, L)]]], device=self.device),
            )
            q_seq = torch.cat([q_seq, next_q], dim=1)

        return anchors

    def predict(self, text: str) -> dict:
        text = text.strip()
        if not text:
            return {"error": "输入文本不能为空"}

        segments = split_with_global_offsets(text, max_chunk_len=55)
        global_anchors: list[dict] = []
        for seg in segments:
            g0 = seg["global_start"]
            for a in self._run_segment(seg["text"]):
                global_anchors.append({
                    "step": len(global_anchors) + 1,
                    "category": a["category"],
                    "color": a["color"],
                    "confidence": a["confidence"],
                    "s0": g0 + a["local_s0"],
                    "e0": g0 + a["local_e0"],
                    "next_action": a["next_action"],
                })

        return {
            "text": text,
            "num_segments": len(segments),
            "num_anchors": len(global_anchors),
            "anchors": global_anchors,
        }


def render_highlighted_text(text: str, anchors: list[dict]) -> str:
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


def show(predictor: SentimentSlicePredictor, line: str):
    res = predictor.predict(line)
    highlighted = render_highlighted_text(res["text"], res["anchors"])
    print("\n" + "=" * 65)
    print(f"输入文本:     {res['text']}")
    print(f"情绪高亮:     {highlighted}")
    print(f"触发切片数:   {res['num_anchors']} 个")
    if res["num_anchors"] == 0:
        print("情绪详情:     中性 / 无明显情绪倾向")
    else:
        for a in res["anchors"]:
            snip = res["text"][a["s0"]:a["e0"] + 1]
            print(f"  Step {a['step']}: {a['color']}[{a['s0']+1}:{a['e0']+1}]{RESET} "
                  f"{a['category']:10s} 置信度={a['confidence']:.3f} '{snip}'")
    print("=" * 65)


def main():
    ap = argparse.ArgumentParser(description="DTSeek 情绪切片分类与高亮演示")
    ap.add_argument("text", nargs="?", help="待检测文本；留空进入交互模式")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    args = ap.parse_args()

    predictor = SentimentSlicePredictor(ckpt_path=args.ckpt)

    if args.text:
        show(predictor, args.text)
        return

    print("\n" + "=" * 65)
    print("  DTSeek 对话情绪切片分类演示（NanoDocEncoder 基座 + 187 词情绪词典）")
    print("  可识别: 积极喜悦(绿) / 愤怒不满(红) / 悲伤焦虑(蓝) / 中性客观(灰)")
    print("  输入 exit 或 quit 退出")
    print("=" * 65)
    while True:
        try:
            line = input("\nDTSeek-Sentiment > ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                break
            show(predictor, line)
        except (KeyboardInterrupt, EOFError):
            print()
            break


if __name__ == "__main__":
    main()
