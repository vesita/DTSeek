"""DTSeek 增强型自回归切片生成与长文档高亮演示入口。

核心特性：
1. 标点感知自适应分句引擎：
   自动将数千字的长篇大论拆解为短句，同时无损追踪全局 1-based 字符坐标；
2. 保持最优感受野：
   确保每个切片检测任务均在小模型的敏锐注意力窗口内（<= 64 字符）执行，彻底消灭截断丢失与位置编码退化；
3. 全局字符级高亮呈现：
   聚合各子句检测出的切片锚点，在完整长文本上进行终端 ANSI 下划线与彩色高亮渲染。
"""
import argparse
import os
import torch
import torch.nn.functional as F

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.doc_encoder import SimpleDocEncoder
from dtseek.robust_ar_model import RobustARSliceDecoder
from dtseek.segmenter import split_with_global_offsets

CLASSES = [
    {"id": 0, "name": "无代词(背景)", "color": "\033[90m"},      # 灰色
    {"id": 1, "name": "第一人称",    "color": "\033[1;36m"},    # 青色加粗
    {"id": 2, "name": "第二人称",    "color": "\033[1;32m"},    # 绿色加粗
    {"id": 3, "name": "第三人称",    "color": "\033[1;33m"},    # 黄色加粗
]
RESET = "\033[0m"


class LongDocARSlicePredictor:
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

    def _predict_segment(self, segment_text: str, max_steps: int = 5) -> list:
        """Predicts slices for a single short segment (<= 64 chars)."""
        enc = self.tokenizer.encode(segment_text, max_length=64, padding=True)
        inp = torch.tensor([enc["input_ids"]], device=self.device)
        mask = torch.tensor([enc["attention_mask"]], dtype=torch.bool, device=self.device)
        L = len(segment_text)

        with torch.no_grad():
            doc_memory = self.doc_encoder(inp, attention_mask=mask)
            q_seq = self.decoder.bos_query.clone()

            anchors = []
            seen_spans = set()

            for step in range(max_steps):
                step_out = self.decoder.forward_step(q_seq, doc_memory, doc_mask=mask)
                cls_prob = F.softmax(step_out["cls_logits"][0], dim=-1)
                pred_cls = int(torch.argmax(cls_prob).item())

                start_idx = int(torch.argmax(step_out["start_logits"][0]).item())
                end_idx = int(torch.argmax(step_out["end_logits"][0]).item())

                act_prob = F.softmax(step_out["action_logits"][0], dim=-1)
                action = int(torch.argmax(act_prob).item())

                if pred_cls == 0:
                    break

                s0 = max(0, min(L - 1, min(start_idx, end_idx)))
                e0 = max(s0, min(L - 1, max(start_idx, end_idx)))

                if (s0, e0) in seen_spans:
                    break
                seen_spans.add((s0, e0))

                anchors.append({
                    "category": CLASSES[pred_cls]["name"],
                    "category_id": pred_cls,
                    "color": CLASSES[pred_cls]["color"],
                    "confidence": round(float(cls_prob[pred_cls].item()), 4),
                    "local_s0": s0,
                    "local_e0": e0,
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

        return anchors

    def predict_long_text(self, text: str) -> dict:
        text = text.strip()
        if not text:
            return {"error": "输入文本不能为空"}

        # 1. Segment document into natural clauses while preserving exact global offsets
        segments = split_with_global_offsets(text, max_chunk_len=55)

        global_anchors = []
        step_counter = 1

        for seg in segments:
            seg_text = seg["text"]
            g_start = seg["global_start"]

            seg_anchors = self._predict_segment(seg_text)
            for a in seg_anchors:
                # Map to global 0-based and 1-based closed intervals
                global_s0 = g_start + a["local_s0"]
                global_e0 = g_start + a["local_e0"]
                start_1 = global_s0 + 1
                end_1 = global_e0 + 1

                global_anchors.append({
                    "step": step_counter,
                    "category": a["category"],
                    "category_id": a["category_id"],
                    "color": a["color"],
                    "confidence": a["confidence"],
                    "span": [start_1, end_1],
                    "s0": global_s0,
                    "e0": global_e0,
                    "next_action": a["next_action"],
                })
                step_counter += 1

        return {
            "text": text,
            "num_segments": len(segments),
            "num_anchors": len(global_anchors),
            "anchors": global_anchors,
        }


def render_highlighted_text(text: str, anchors: list) -> str:
    if not anchors:
        return text

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
                rendered.append(f"\033[4m{style}")
            current_style = style
        rendered.append(ch)

    if current_style is not None:
        rendered.append(RESET)

    return "".join(rendered)


def main():
    parser = argparse.ArgumentParser(description="DTSeek 长文本自适应语句切片高亮演示")
    parser.add_argument("text", nargs="?", type=str, help="待检测的长短中文文本")
    parser.add_argument("--ckpt", default="checkpoints/robust_ar_dtseek.pt", help="模型权重路径")
    args = parser.parse_args()

    predictor = LongDocARSlicePredictor(ckpt_path=args.ckpt)

    if args.text:
        res = predictor.predict_long_text(args.text)
        highlighted = render_highlighted_text(res["text"], res["anchors"])
        print("\n" + "=" * 65)
        print(f"输入文本长度:   {len(res['text'])} 字符 (自动拆分为 {res['num_segments']} 个自适应语义分句)")
        print(f"触发锚点总数:   {res['num_anchors']} 个")
        print("\n[切片高亮正文 (全文无截断渲染)]:\n")
        print(highlighted)
        print("\n" + "-" * 65)
        print("模型吐出锚点详情 (Global 1-Based Anchors):")
        for a in res["anchors"]:
            print(f"  Step {a['step']:2d}: {a['color']}[{a['span'][0]:3d}:{a['span'][1]:3d}]{RESET} | {a['category']:10s} | 置信度: {a['confidence']:.3f} | 原文: '{res['text'][a['s0']:a['e0']+1]}'")
        print("=" * 65 + "\n")
        return

    print("\n" + "=" * 65)
    print("  DTSeek 长文本自适应分句切片高亮演示")
    print("  支持任意千字长篇大论，自动标点切分子句 + 全局精准坐标映射。")
    print("  输入 'exit' 或 'quit' 退出。")
    print("=" * 65 + "\n")

    while True:
        try:
            line = input("DTSeek > ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                break
            res = predictor.predict_long_text(line)
            highlighted = render_highlighted_text(res["text"], res["anchors"])
            print(f" -> 高亮解析: {highlighted}")
            if res["num_anchors"] == 0:
                print("    (纯背景文本)")
            else:
                for a in res["anchors"]:
                    print(f"    • Step {a['step']}: {a['color']}[{a['span'][0]}:{a['span'][1]}]{RESET} ({a['category']}, 置信度: {a['confidence']}) '{res['text'][a['s0']:a['e0']+1]}'")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n退出。")
            break


if __name__ == "__main__":
    main()
