"""Interactive and CLI demo for YOLO-style [Localization Span + Category + Confidence] Decision Engine."""
import argparse
import os
import torch
import torch.nn.functional as F

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.doc_encoder import SimpleDocEncoder
from dtseek.model import DTSeekConfig, DTSeekModel
from dtseek.query_projector import TextGuidedQueryProjector

CLASSES = [
    {"id": 0, "name": "无代词(背景)", "desc": "不包含明显人称代词"},
    {"id": 1, "name": "第一人称", "desc": "我、我们、咱们、俺、鄙人"},
    {"id": 2, "name": "第二人称", "desc": "你、你们、您、阁下"},
    {"id": 3, "name": "第三人称", "desc": "他、她、它、他们、她们、它们"},
]


class YOLODecisionPredictor:
    def __init__(self, ckpt_path: str = "checkpoints/yolo_dtseek.pt", device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = NanoCharTokenizer()

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.config = ckpt["config"]
        hidden_dim = self.config.hidden_dim

        self.doc_encoder = SimpleDocEncoder(
            vocab_size=self.tokenizer.vocab_size,
            hidden_dim=hidden_dim,
            num_layers=3,
            num_heads=4,
        ).to(self.device)
        self.doc_encoder.load_state_dict(ckpt["doc_encoder"])
        self.doc_encoder.eval()

        self.model = DTSeekModel(self.config).to(self.device)
        self.model.load_state_dict(ckpt["dtseek"])
        self.model.eval()

        self.query_proj = TextGuidedQueryProjector(
            hidden_dim=hidden_dim,
            num_classes=4,
            use_background_class=False,
        ).to(self.device)
        self.query_proj.load_state_dict(ckpt["query_proj"])
        self.query_proj.eval()

    def predict(self, text: str) -> dict:
        text = text.strip()
        if not text:
            return {"error": "输入文本不能为空"}

        enc = self.tokenizer.encode(text, max_length=64, padding=True)
        inp = torch.tensor([enc["input_ids"]], device=self.device)
        mask = torch.tensor([enc["attention_mask"]], dtype=torch.bool, device=self.device)
        L = len(text)

        with torch.no_grad():
            doc_memory = self.doc_encoder(inp, attention_mask=mask)
            queries, q_mask = self.query_proj(batch_size=1)
            out = self.model(queries, doc_memory, doc_mask=mask, query_mask=q_mask)

        probs = out["probs"][0].cpu().tolist()
        pred_id = int(out["best_index"][0].item())
        conf = float(out["confidence"][0].item())

        # Extract 1D Bounding Span for the detected class query
        span_bounds = out["span_bounds"][0, pred_id].cpu().tolist()  # [start_norm, end_norm]
        start_char_0 = max(0, min(L - 1, int(round(span_bounds[0] * L))))
        end_char_0 = max(start_char_0 + 1, min(L, int(round(span_bounds[1] * L))))

        if pred_id == 0:
            trigger_span = None
            detected_snippet = ""
        else:
            # 1-based natural closed interval [start_1, end_1]
            trigger_span = [start_char_0 + 1, end_char_0]
            detected_snippet = text[start_char_0:end_char_0]

        prob_breakdown = {CLASSES[i]["name"]: round(probs[i], 4) for i in range(len(CLASSES))}

        return {
            "text": text,
            "prediction": CLASSES[pred_id]["name"],
            "prediction_id": pred_id,
            "confidence": round(conf, 4),
            "probabilities": prob_breakdown,
            "trigger_span": trigger_span,
            "detected_snippet": detected_snippet,
        }


def main():
    parser = argparse.ArgumentParser(description="DTSeek YOLO风格 [定位+类别+置信度] 决策演示")
    parser.add_argument("text", nargs="?", type=str, help="待检测的中文句子")
    parser.add_argument("--ckpt", default="checkpoints/yolo_dtseek.pt", help="模型权重路径")
    args = parser.parse_args()

    predictor = YOLODecisionPredictor(ckpt_path=args.ckpt)

    if args.text:
        res = predictor.predict(args.text)
        print("\n" + "=" * 55)
        print(f"输入文本:   {res['text']}")
        print(f"类别判定:   {res['prediction']}")
        print(f"置信度:     {res['confidence']}")
        if res["trigger_span"]:
            print(f"定位区间:   字符 [{res['trigger_span'][0]}:{res['trigger_span'][1]}] -> '{res['detected_snippet']}'")
        else:
            print("定位区间:   无 (纯背景类)")
        print("概率分布:")
        for k, v in res["probabilities"].items():
            bar = "█" * int(v * 25)
            print(f"  {k:12s} : {v:.4f} {bar}")
        print("=" * 55 + "\n")
        return

    print("\n" + "=" * 65)
    print("  DTSeek (YOLO-Style [定位+类别+置信度]) 决策引擎演示")
    print("  输入任意中文句子，模型将同时输出：类别、触发词定位区间与置信度。")
    print("  输入 'exit' 或 'quit' 退出。")
    print("=" * 65 + "\n")

    while True:
        try:
            line = input("DTSeek-YOLO > ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                break
            res = predictor.predict(line)
            span_info = f" | 定位: [{res['trigger_span'][0]}:{res['trigger_span'][1]}] '{res['detected_snippet']}'" if res["trigger_span"] else ""
            print(f" -> 结果: \033[1;32m{res['prediction']}\033[0m (置信度: {res['confidence']}){span_info}")
            print("    分布: ", end="")
            for k, v in res["probabilities"].items():
                print(f"{k}: {v*100:.1f}%  ", end="")
            print("\n")
        except (KeyboardInterrupt, EOFError):
            print("\n退出。")
            break


if __name__ == "__main__":
    main()
