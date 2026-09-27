"""Interactive and CLI demo for Robust Autoregressive Slice Emission with Dynamic Text Highlighting.

Core Paradigm:
- Model output: purely discrete anchor indices [start_1, end_1] + category_id + confidence + action.
  (NO raw string output from model, zero text hallucination).
- Engine presentation: renders original input with beautiful terminal ANSI color highlights based on anchor bounds!
"""
import argparse
import os
import torch
import torch.nn.functional as F

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.doc_encoder import SimpleDocEncoder
from dtseek.robust_ar_model import RobustARSliceDecoder

CLASSES = [
    {"id": 0, "name": "无代词(背景)", "color": "\033[90m"},      # 灰色
    {"id": 1, "name": "第一人称",    "color": "\033[1;36m"},    # 青色加粗
    {"id": 2, "name": "第二人称",    "color": "\033[1;32m"},    # 绿色加粗
    {"id": 3, "name": "第三人称",    "color": "\033[1;33m"},    # 黄色加粗
]
RESET = "\033[0m"


class RobustARSlicePredictor:
    def __init__(self, ckpt_path: str = "checkpoints/robust_ar_dtseek.pt", device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = NanoCharTokenizer()

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        hidden_dim = ckpt["hidden_dim"]

        self.doc_encoder = SimpleDocEncoder(
            vocab_size=self.tokenizer.vocab_size,
            hidden_dim=hidden_dim,
            num_layers=3,
            num_heads=4,
        ).to(self.device)
        self.doc_encoder.load_state_dict(ckpt["doc_encoder"])
        self.doc_encoder.eval()

        self.decoder = RobustARSliceDecoder(
            hidden_dim=hidden_dim,
            num_classes=4,
            num_heads=4,
            num_layers=2,
        ).to(self.device)
        self.decoder.load_state_dict(ckpt["decoder"])
        self.decoder.eval()

    def predict_anchors(self, text: str, max_steps: int = 6) -> dict:
        """Returns purely structural anchor outputs: start/end 1-based indices, categories, confidence, next_action."""
        text = text.strip()
        if not text:
            return {"error": "输入文本不能为空"}

        enc = self.tokenizer.encode(text, max_length=64, padding=True)
        inp = torch.tensor([enc["input_ids"]], device=self.device)
        mask = torch.tensor([enc["attention_mask"]], dtype=torch.bool, device=self.device)
        L = len(text)

        with torch.no_grad():
            doc_memory = self.doc_encoder(inp, attention_mask=mask)
            q_seq = self.decoder.bos_query.clone()  # [1, 1, D]

            anchors = []
            seen_spans = set()

            for step in range(max_steps):
                step_out = self.decoder.forward_step(q_seq, doc_memory, doc_mask=mask)

                cls_prob = F.softmax(step_out["cls_logits"][0], dim=-1)
                pred_cls = int(torch.argmax(cls_prob).item())

                start_idx = int(torch.argmax(step_out["start_logits"][0]).item())
                end_idx = int(torch.argmax(step_out["end_logits"][0]).item())

                act_prob = F.softmax(step_out["action_logits"][0], dim=-1)
                action = int(torch.argmax(act_prob).item())  # 0: <eos>, 1: <cont>

                if pred_cls == 0:
                    break

                s0 = max(0, min(L - 1, min(start_idx, end_idx)))
                e0 = max(s0, min(L - 1, max(start_idx, end_idx)))

                span_tuple = (s0, e0)
                if span_tuple in seen_spans:
                    break
                seen_spans.add(span_tuple)

                # 1-based closed interval [start_1, end_1]
                start_1 = s0 + 1
                end_1 = e0 + 1

                anchors.append({
                    "step": step + 1,
                    "category": CLASSES[pred_cls]["name"],
                    "category_id": pred_cls,
                    "color": CLASSES[pred_cls]["color"],
                    "confidence": round(float(cls_prob[pred_cls].item()), 4),
                    "span": [start_1, end_1],
                    "s0": s0,
                    "e0": e0,
                    "next_action": "<cont>" if action == 1 else "<eos>",
                })

                if action == 0:
                    break

                norm_s = torch.tensor([[[s0 / max(1, L)]]], device=self.device)
                norm_e = torch.tensor([[[e0 / max(1, L)]]], device=self.device)
                prev_c = torch.tensor([[pred_cls]], device=self.device)

                next_q = self.decoder.get_step_input(
                    prev_hidden=step_out["last_hidden"],
                    prev_cls=prev_c,
                    prev_start=norm_s,
                    prev_end=norm_e,
                )
                q_seq = torch.cat([q_seq, next_q], dim=1)

        return {
            "text": text,
            "num_anchors": len(anchors),
            "anchors": anchors,
        }


def render_highlighted_text(text: str, anchors: list) -> str:
    """Takes original input text and overlays ANSI color tags based strictly on anchor index spans."""
    if not anchors:
        return text

    # Map each character index to a color tag if within an anchor
    char_styles = [None] * len(text)
    for a in anchors:
        for i in range(a["s0"], a["e0"] + 1):
            if i < len(text):
                char_styles[i] = a["color"]

    rendered = []
    current_style = None
    for i, ch in enumerate(text):
        style = char_styles[i]
        if style != current_style:
            if current_style is not None:
                rendered.append(RESET)
            if style is not None:
                rendered.append(f"\033[4m{style}")  # Bold Color + Underline
            current_style = style
        rendered.append(ch)

    if current_style is not None:
        rendered.append(RESET)

    return "".join(rendered)


def main():
    parser = argparse.ArgumentParser(description="DTSeek 语句切片高亮输出演示")
    parser.add_argument("text", nargs="?", type=str, help="待检测的中文句子")
    parser.add_argument("--ckpt", default="checkpoints/robust_ar_dtseek.pt", help="模型权重路径")
    args = parser.parse_args()

    predictor = RobustARSlicePredictor(ckpt_path=args.ckpt)

    if args.text:
        res = predictor.predict_anchors(args.text)
        highlighted = render_highlighted_text(res["text"], res["anchors"])
        print("\n" + "=" * 65)
        print(f"原始输入:       {res['text']}")
        print(f"切片高亮:       {highlighted}")
        print(f"触发锚点数:     {res['num_anchors']} 个")
        if res["num_anchors"] == 0:
            print("锚点详情:       无 (模型首步直接预测 <eos>，纯背景文本)")
        else:
            print("模型吐出锚点 (Pure Position Anchors):")
            for a in res["anchors"]:
                act_tag = f"-> 动作: {a['next_action']}"
                print(f"  Step {a['step']}: {a['color']}[{a['span'][0]}:{a['span'][1]}]{RESET} | {a['category']:10s} | 置信度: {a['confidence']:.3f} | {act_tag}")
        print("=" * 65 + "\n")
        return

    print("\n" + "=" * 65)
    print("  DTSeek 语句切片高亮显示演示 (纯锚点定位 + 终端着色)")
    print("  模型仅吐出数字锚点与类别，由界面高亮原文中对应的字符。")
    print("  输入 'exit' 或 'quit' 退出。")
    print("=" * 65 + "\n")

    while True:
        try:
            line = input("DTSeek > ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                break
            res = predictor.predict_anchors(line)
            highlighted = render_highlighted_text(res["text"], res["anchors"])
            print(f" -> 高亮解析: {highlighted}")
            if res["num_anchors"] == 0:
                print("    (纯背景文本，首步直接 <eos>)")
            else:
                for a in res["anchors"]:
                    print(f"    • Step {a['step']}: {a['color']}[{a['span'][0]}:{a['span'][1]}]{RESET} ({a['category']}, 置信度: {a['confidence']}) {a['next_action']}")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n退出。")
            break


if __name__ == "__main__":
    main()
