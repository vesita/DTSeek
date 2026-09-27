"""Interactive and CLI demo for Autoregressive Slice Emission (<slice>...<cont>/<eos>).

Mechanism:
1. Doc Encoder encodes document memory [1, L, D]
2. Step 1: Decode first slice (category + start/end pointer) and action (<cont> vs <eos>).
3. If <cont>: Append previous slice representation into query history, re-attend to Doc Memory, emit next slice!
4. If <eos>: Terminate sequence.
Supports arbitrary number of slices dynamically!
"""
import argparse
import os
import torch
import torch.nn.functional as F

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.doc_encoder import SimpleDocEncoder
from dtseek.ar_slice_model import AutoregressiveSliceDecoder

CLASSES = [
    {"id": 0, "name": "无代词(背景)"},
    {"id": 1, "name": "第一人称"},
    {"id": 2, "name": "第二人称"},
    {"id": 3, "name": "第三人称"},
]


class ARSlicePredictor:
    def __init__(self, ckpt_path: str = "checkpoints/ar_slice_dtseek.pt", device: str = None):
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

        self.decoder = AutoregressiveSliceDecoder(
            hidden_dim=hidden_dim,
            num_classes=4,
            num_heads=4,
            num_layers=2,
        ).to(self.device)
        self.decoder.load_state_dict(ckpt["decoder"])
        self.decoder.eval()

    def predict(self, text: str, max_steps: int = 6) -> dict:
        text = text.strip()
        if not text:
            return {"error": "输入文本不能为空"}

        enc = self.tokenizer.encode(text, max_length=64, padding=True)
        inp = torch.tensor([enc["input_ids"]], device=self.device)
        mask = torch.tensor([enc["attention_mask"]], dtype=torch.bool, device=self.device)
        L = len(text)

        with torch.no_grad():
            doc_memory = self.doc_encoder(inp, attention_mask=mask)
            q_hist = self.decoder.bos_query.clone()  # [1, 1, D]

            slices = []
            for step in range(max_steps):
                step_out = self.decoder.forward_step(q_hist, doc_memory, doc_mask=mask)

                cls_prob = F.softmax(step_out["cls_logits"][0], dim=-1)
                pred_cls = int(torch.argmax(cls_prob).item())

                start_idx = int(torch.argmax(step_out["start_logits"][0]).item())
                end_idx = int(torch.argmax(step_out["end_logits"][0]).item())

                action_prob = F.softmax(step_out["action_logits"][0], dim=-1)
                action = int(torch.argmax(action_prob).item())  # 0: <eos>, 1: <cont>

                # If first step predicts background (0) or <eos> with no content, stop
                if pred_cls == 0:
                    break

                # Ensure 1-based closed interval [start_1, end_1]
                s0 = max(0, min(L - 1, min(start_idx, end_idx)))
                e0 = max(s0, min(L - 1, max(start_idx, end_idx)))

                start_1 = s0 + 1
                end_1 = e0 + 1
                snippet = text[s0:e0+1]

                slices.append({
                    "step": step + 1,
                    "category": CLASSES[pred_cls]["name"],
                    "category_id": pred_cls,
                    "confidence": round(float(cls_prob[pred_cls].item()), 4),
                    "slice_span": [start_1, end_1],
                    "slice_text": snippet,
                    "next_action": "<cont>" if action == 1 else "<eos>",
                })

                if action == 0:  # <eos> emitted, end of slices!
                    break

                # Append step representation back into query history to condition the NEXT slice!
                q_hist = torch.cat([q_hist, step_out["last_hidden"]], dim=1)

        return {
            "text": text,
            "num_slices": len(slices),
            "slices": slices,
        }


def main():
    parser = argparse.ArgumentParser(description="DTSeek 自回归切片序列生成 (<slice>...<cont>/<eos>)")
    parser.add_argument("text", nargs="?", type=str, help="待检测的中文句子")
    parser.add_argument("--ckpt", default="checkpoints/ar_slice_dtseek.pt", help="模型权重路径")
    args = parser.parse_args()

    predictor = ARSlicePredictor(ckpt_path=args.ckpt)

    if args.text:
        res = predictor.predict(args.text)
        print("\n" + "=" * 65)
        print(f"输入文本:       {res['text']}")
        print(f"检测到切片数:   {res['num_slices']} 个")
        if res["num_slices"] == 0:
            print("切片列表:       无 (模型首步直接预测 <eos>，纯背景文本)")
        else:
            print("切片发射序列 (Autoregressive Steps):")
            for s in res["slices"]:
                act_tag = f"-> 动作: {s['next_action']}"
                print(f"  Step {s['step']}: [{s['slice_span'][0]}:{s['slice_span'][1]}] '{s['slice_text']}' | 类别: {s['category']:10s} | 置信度: {s['confidence']:.3f} | {act_tag}")
        print("=" * 65 + "\n")
        return

    print("\n" + "=" * 65)
    print("  DTSeek 自回归切片生成 (<slice>...<cont>/<eos>) 演示")
    print("  每次吐出一个锚点，根据 <cont>/<eos> 决定是否回填并吐出下一个。")
    print("  输入 'exit' 或 'quit' 退出。")
    print("=" * 65 + "\n")

    while True:
        try:
            line = input("DTSeek-AR > ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                break
            res = predictor.predict(line)
            print(f" -> 吐出 {res['num_slices']} 个切片:")
            if res["num_slices"] == 0:
                print("    (直接 <eos>，无目标切片)")
            else:
                for s in res["slices"]:
                    print(f"    • Step {s['step']}: [{s['slice_span'][0]}:{s['slice_span'][1]}] '\033[1;32m{s['slice_text']}\033[0m' ({s['category']}, 置信度: {s['confidence']}) {s['next_action']}")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n退出。")
            break


if __name__ == "__main__":
    main()
