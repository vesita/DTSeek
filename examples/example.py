"""Interactive and CLI evaluation example for DTSeek pronoun decision model."""
import argparse
import os
import torch

from dtseek.tokenizer import NanoCharTokenizer
from dtseek.doc_encoder import SimpleDocEncoder
from dtseek.model import DTSeekModel


CLASSES = [
    {"id": 0, "name": "无代词(背景)", "desc": "句子不包含明显人称代词"},
    {"id": 1, "name": "第一人称", "desc": "包含 我、我们、咱们、俺 等"},
    {"id": 2, "name": "第二人称", "desc": "包含 你、你们、您 等"},
    {"id": 3, "name": "第三人称", "desc": "包含 他、她、它、他们 等"},
]


class PronounPredictor:
    def __init__(self, ckpt_path: str = "checkpoints/pronoun_dtseek.pt", device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = NanoCharTokenizer()

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Please run `uv run python train_pronoun.py` first.")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.config = ckpt["config"]
        hidden_dim = self.config.hidden_dim

        # Load Doc Encoder
        self.doc_encoder = SimpleDocEncoder(
            vocab_size=self.tokenizer.vocab_size,
            hidden_dim=hidden_dim,
            num_layers=3,
            num_heads=4,
        ).to(self.device)
        self.doc_encoder.load_state_dict(ckpt["doc_encoder"])
        self.doc_encoder.eval()

        # Load DTSeek Model (support both pure classification and span slice classification)
        self.model = DTSeekModel(self.config).to(self.device)
        self.model.load_state_dict(ckpt["dtseek"], strict=False)
        self.model.eval()

        self.has_span_head = ("span_head.0.weight" in ckpt["dtseek"])

        # Handle backward compatibility: Query Projector or raw class_queries
        if "query_proj" in ckpt:
            from dtseek.query_projector import TextGuidedQueryProjector
            self.query_proj = TextGuidedQueryProjector(
                hidden_dim=hidden_dim,
                num_classes=len(CLASSES),
                use_background_class=False,
            ).to(self.device)
            self.query_proj.load_state_dict(ckpt["query_proj"])
            self.query_proj.eval()
            self.class_queries = None
        else:
            self.query_proj = None
            self.class_queries = ckpt["class_queries"].to(self.device)

    def predict(self, text: str) -> dict:
        text = text.strip()
        if not text:
            return {"error": "输入文本不能为空"}

        enc = self.tokenizer.encode(text, max_length=64, padding=True)
        inp = torch.tensor([enc["input_ids"]], device=self.device)
        mask = torch.tensor([enc["attention_mask"]], dtype=torch.bool, device=self.device)

        with torch.no_grad():
            doc_memory = self.doc_encoder(inp, attention_mask=mask)
            if self.query_proj is not None:
                queries, q_mask = self.query_proj(batch_size=1)
                q = queries
                for layer in self.model.decoder_layers:
                    q = layer(q, doc_memory, doc_mask=mask, query_mask=q_mask)
                q = self.model.final_norm(q)
                logits = self.model.cat_scorer(q).squeeze(-1)
                probs = torch.softmax(logits, dim=-1)
                top2 = probs.topk(2, dim=-1).values
                conf = float((top2[:, 0] - top2[:, 1])[0].item())
                pred_id = int(torch.argmax(logits, dim=-1)[0].item())
                probs_list = probs[0].cpu().tolist()
                is_bg = (pred_id == 0)
            else:
                out = self.model(self.class_queries, doc_memory, doc_mask=mask)
                probs_list = out["probs"][0].cpu().tolist()
                pred_id = int(out["best_index"][0].item())
                conf = float(out["confidence"][0].item())
                is_bg = bool(out["is_background"][0].item())

        # 语句切片分类（Span Slice Detection）: 提取 1-based 闭区间 [start, end]
        L = len(text)
        if self.has_span_head and pred_id != 0:
            with torch.no_grad():
                spans = self.model.span_head(q)  # [1, TotalQueries, 2]
                center = spans[0, pred_id, 0].item()
                width = spans[0, pred_id, 1].item()
                start_0 = max(0, min(L - 1, int(round((center - width / 2.0) * L))))
                end_0 = max(start_0 + 1, min(L, int(round((center + width / 2.0) * L))))
                slice_span = [start_0 + 1, end_0]  # 1-based 闭区间
                slice_text = text[start_0:end_0]
        else:
            slice_span = None
            slice_text = ""

        prob_breakdown = {CLASSES[i]["name"]: round(probs_list[i], 4) for i in range(len(CLASSES))}

        return {
            "text": text,
            "prediction": CLASSES[pred_id]["name"],
            "prediction_id": pred_id,
            "is_background": is_bg,
            "confidence": round(conf, 4),
            "slice_span": slice_span,
            "slice_text": slice_text,
            "probabilities": prob_breakdown,
        }


def main():
    parser = argparse.ArgumentParser(description="DTSeek 代词判定示例")
    parser.add_argument("text", nargs="?", type=str, help="待检测的中文句子 (如果留空则进入交互模式)")
    parser.add_argument("--ckpt", default="checkpoints/pronoun_dtseek.pt", help="模型权重路径")
    args = parser.parse_args()

    predictor = PronounPredictor(ckpt_path=args.ckpt)

    # 1. 单次命令行模式
    if args.text:
        res = predictor.predict(args.text)
        print("\n" + "=" * 55)
        print(f"输入文本:       {res['text']}")
        print(f"判定结果:       {res['prediction']}")
        print(f"置信度:         {res['confidence']}")
        if res["slice_span"]:
            print(f"语句切片定位:   [{res['slice_span'][0]}:{res['slice_span'][1]}] (1-based闭区间) -> '{res['slice_text']}'")
        print("类别概率分布:")
        for k, v in res["probabilities"].items():
            bar = "█" * int(v * 30)
            print(f"  {k:12s} : {v:.4f} {bar}")
        print("=" * 55 + "\n")
        return

    # 2. 交互式命令行模式 (REPL)
    print("\n" + "=" * 65)
    print("  欢迎使用 DTSeek 语句切片分类 (Span Slice Classification) 演示")
    print("  请输入任意中文句子，模型将同时输出类别判定与 1-based 闭区间切片定位。")
    print("  输入 'exit' 或 'quit' 退出。")
    print("=" * 65 + "\n")

    while True:
        try:
            line = input("DTSeek > ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                print("退出。")
                break
            res = predictor.predict(line)
            slice_info = f" | 切片定位: [{res['slice_span'][0]}:{res['slice_span'][1]}] '{res['slice_text']}'" if res["slice_span"] else ""
            print(f" -> 结果: \033[1;32m{res['prediction']}\033[0m (置信度: {res['confidence']}){slice_info}")
            print("    概率分布: ", end="")
            for k, v in res["probabilities"].items():
                print(f"{k}: {v*100:.1f}%  ", end="")
            print("\n")
        except (KeyboardInterrupt, EOFError):
            print("\n退出。")
            break


if __name__ == "__main__":
    main()
