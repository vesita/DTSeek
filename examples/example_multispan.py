"""Interactive and CLI demo for Multi-Span Slice Classification (DETR Slot Paradigm).

Outputs variable number of detected slices per sentence:
- Each detected slice includes:
  - 1-based closed interval [start, end]
  - Category (第一人称, 第二人称, 第三人称)
  - Objectness confidence score
  - Extracted snippet text
"""
import argparse
import os
import torch
import torch.nn.functional as F

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.doc_encoder import SimpleDocEncoder
from dtseek.slot_detector import SpanSlotDecoder

CLASSES = [
    {"id": 0, "name": "背景/无切片"},
    {"id": 1, "name": "第一人称"},
    {"id": 2, "name": "第二人称"},
    {"id": 3, "name": "第三人称"},
]


class MultiSpanPredictor:
    def __init__(self, ckpt_path: str = "checkpoints/slot_multispan_dtseek.pt", device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = NanoCharTokenizer()

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        hidden_dim = ckpt["hidden_dim"]
        num_slots = ckpt.get("num_slots", 8)

        self.doc_encoder = SimpleDocEncoder(
            vocab_size=self.tokenizer.vocab_size,
            hidden_dim=hidden_dim,
            num_layers=3,
            num_heads=4,
        ).to(self.device)
        self.doc_encoder.load_state_dict(ckpt["doc_encoder"])
        self.doc_encoder.eval()

        self.slot_decoder = SpanSlotDecoder(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_slots=num_slots,
            num_classes=4,
            num_layers=2,
        ).to(self.device)
        self.slot_decoder.load_state_dict(ckpt["slot_decoder"])
        self.slot_decoder.eval()

    def predict(self, text: str, conf_threshold: float = 0.5) -> dict:
        text = text.strip()
        if not text:
            return {"error": "输入文本不能为空"}

        enc = self.tokenizer.encode(text, max_length=64, padding=True)
        inp = torch.tensor([enc["input_ids"]], device=self.device)
        mask = torch.tensor([enc["attention_mask"]], dtype=torch.bool, device=self.device)
        L = len(text)

        with torch.no_grad():
            doc_memory = self.doc_encoder(inp, attention_mask=mask)
            out = self.slot_decoder(doc_memory, doc_mask=mask)

        probs = out["probs"][0]          # [K, 4]
        span_bounds = out["span_bounds"][0]  # [K, 2] (start, end)
        confidence = out["confidence"][0]    # [K]

        detected_slices = []
        K = probs.shape[0]

        for k in range(K):
            pred_cls = int(torch.argmax(probs[k]).item())
            cls_prob = float(probs[k, pred_cls].item())
            slot_conf = float(confidence[k].item())

            # Only retain slots predicted as positive non-background categories with valid confidence
            if pred_cls > 0 and (slot_conf >= conf_threshold or cls_prob >= conf_threshold):
                s_norm = span_bounds[k, 0].item()
                e_norm = span_bounds[k, 1].item()

                start_0 = max(0, min(L - 1, int(round(s_norm * L))))
                end_0 = max(start_0 + 1, min(L, int(round(e_norm * L))))

                start_1 = start_0 + 1
                end_1 = end_0
                snippet = text[start_0:end_0]

                detected_slices.append({
                    "slot_id": k,
                    "category": CLASSES[pred_cls]["name"],
                    "category_id": pred_cls,
                    "confidence": round(cls_prob * slot_conf, 4),
                    "raw_confidence": round(slot_conf, 4),
                    "class_prob": round(cls_prob, 4),
                    "slice_span": [start_1, end_1],
                    "slice_text": snippet,
                })

        # Sort slices by natural start position in text
        detected_slices.sort(key=lambda x: x["slice_span"][0])

        return {
            "text": text,
            "num_slices": len(detected_slices),
            "slices": detected_slices,
        }


def main():
    parser = argparse.ArgumentParser(description="DTSeek 动态多切片语句分类演示")
    parser.add_argument("text", nargs="?", type=str, help="待检测的中文句子")
    parser.add_argument("--ckpt", default="checkpoints/slot_multispan_dtseek.pt", help="模型权重路径")
    parser.add_argument("--thresh", default=0.5, type=float, help="切片置信度过滤阈值")
    args = parser.parse_args()

    predictor = MultiSpanPredictor(ckpt_path=args.ckpt)

    if args.text:
        res = predictor.predict(args.text, conf_threshold=args.thresh)
        print("\n" + "=" * 60)
        print(f"输入文本:       {res['text']}")
        print(f"检测到切片数:   {res['num_slices']} 个")
        if res["num_slices"] == 0:
            print("切片列表:       无 (句子中未检测到目标代词切片)")
        else:
            print("切片详情:")
            for i, s in enumerate(res["slices"], 1):
                print(f"  [{i}] 切片区间: [{s['slice_span'][0]}:{s['slice_span'][1]}] -> '{s['slice_text']}' | 类别: {s['category']:10s} | 置信度: {s['confidence']:.3f}")
        print("=" * 60 + "\n")
        return

    print("\n" + "=" * 65)
    print("  DTSeek (DETR-Slot 动态多切片语句分类) 演示")
    print("  可识别单句中【任意可变数量】的切片位置、类别与置信度。")
    print("  输入 'exit' 或 'quit' 退出。")
    print("=" * 65 + "\n")

    while True:
        try:
            line = input("DTSeek-MultiSpan > ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                break
            res = predictor.predict(line, conf_threshold=args.thresh)
            print(f" -> 检测到 {res['num_slices']} 个切片:")
            if res["num_slices"] == 0:
                print("    (纯背景文本)")
            else:
                for s in res["slices"]:
                    print(f"    • [{s['slice_span'][0]}:{s['slice_span'][1]}] '\033[1;32m{s['slice_text']}\033[0m' ({s['category']}, 置信度: {s['confidence']})")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n退出。")
            break


if __name__ == "__main__":
    main()
