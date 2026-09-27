r"""情绪任务词级探针：对词典里每个情绪词逐个测类别是否判对。

为什么需要这个探针：
    句级验证集准确率 98% 会掩盖"某些词系统性判错"——因为高频词样本多、低频词样本少。
    把每个词单独放进中性载体句里测，才能暴露"某个词从来没学会"。

用法：
    uv run python scripts/eval_emotion_words.py
    uv run python scripts/eval_emotion_words.py --ckpt checkpoints/multitask_v2_dtseek.pt
"""
import argparse
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.nano_doc_encoder import NanoDocEncoder
from dtseek.robust_ar_model import RobustARSliceDecoder
from dtseek.sentiment_dataset import LEXICON_BY_CAT

CAT_NAME = {0: "中性/未触发", 1: "积极/喜悦", 2: "愤怒/不满", 3: "悲伤/焦虑"}

# 中性载体句：词放在不同位置，并同时覆盖"带终止标点"与"无终止标点"两种输入形态
# （真实聊天输入常常没有句号；只测带标点的形态会漏掉定位截断类缺陷）
CARRIERS = [
    "{e}。",
    "说实话，{e}。",
    "现在就是{e}。",
    "刚看到这个消息，{e}。",
    "{e}",              # 无终止标点
    "真的很{e}",         # 无终止标点
]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden_dim = ckpt["hidden_dim"]

    enc = NanoDocEncoder(vocab_size=NanoCharTokenizer().vocab_size,
                         hidden_dim=hidden_dim, num_layers=3, num_heads=4,
                         max_len=128, dropout=0.0).to(device)
    enc.load_state_dict(ckpt["doc_encoder"])
    enc.eval()

    dec = RobustARSliceDecoder(hidden_dim=hidden_dim, num_classes=4,
                               num_heads=4, num_layers=2).to(device)
    dec.load_state_dict(ckpt["decoders"]["sentiment"])
    dec.eval()
    return enc, dec


@torch.no_grad()
def predict_first(enc, dec, tok, device, text):
    """返回首切片的 (预测类别, 起止 0-based, 类别置信度)。"""
    e = tok.encode(text, max_length=64, padding=True)
    inp = torch.tensor([e["input_ids"]], device=device)
    mask = torch.tensor([e["attention_mask"]], dtype=torch.bool, device=device)
    doc = enc(inp, attention_mask=mask)
    out = dec.forward_step(dec.bos_query, doc, doc_mask=mask)

    probs = F.softmax(out["cls_logits"][0], dim=-1)
    cls = int(probs.argmax().item())
    s = int(out["start_logits"][0].argmax().item())
    en = int(out["end_logits"][0].argmax().item())
    return cls, (min(s, en), max(s, en)), float(probs[cls].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/multitask_v2_dtseek.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"未找到 {args.ckpt}")

    tok = NanoCharTokenizer()
    enc, dec = load_model(args.ckpt, device)

    # ── 已知答案对照（nanoSeek 纪律 5.4）：测量函数先过已知答案的输入 ──
    # 这些长句在 example_sentiment.py 里已人工验证过，探针必须复现同样的结论，
    # 否则说明探针本身写错了，而不是模型错。
    sanity = [
        ("我感觉有点难受", 3), ("我有点喜欢这个颜色", 1),
        ("这也太糟心了", 2), ("今天心情挺好的", 1),
    ]
    print("=" * 70)
    print("已知答案对照（探针自检）")
    print("=" * 70)
    sanity_fail = 0
    for text, want in sanity:
        pred, (s0, e0), conf = predict_first(enc, dec, tok, device, text)
        ok = (pred == want)
        sanity_fail += (not ok)
        print(f"  {'✓' if ok else '✗'} '{text}' 期望={CAT_NAME[want]} 实际={CAT_NAME[pred]} "
              f"({conf:.3f}) 区间='{text[s0:e0+1]}'")
    if sanity_fail:
        print(f"\n⚠ {sanity_fail} 条对照不符：先怀疑探针写错，再怀疑模型。\n")
    else:
        print("\n对照全部通过 ✅ 探针可信\n")

    total = correct = 0
    span_total = span_correct = 0
    per_cat = defaultdict(lambda: [0, 0])          # cat -> [correct, total]
    span_by_cat = defaultdict(lambda: [0, 0])
    wrong_by_cat = defaultdict(list)               # 期望类别 -> [(词, 载体, 预测类别, 定位是否正确)]

    for cat, words in LEXICON_BY_CAT.items():
        for w in words:
            for carrier in CARRIERS:
                text = carrier.format(e=w)
                pred, (s0, e0), conf = predict_first(enc, dec, tok, device, text)

                # 定位是否正确：预测区间是否正好等于该词（类别对但只圈了一半也算错）
                span_ok = (text[s0:e0 + 1] == w)

                total += 1
                span_total += 1
                per_cat[cat][1] += 1
                span_by_cat[cat][1] += 1
                if pred == cat:
                    correct += 1
                    per_cat[cat][0] += 1
                else:
                    wrong_by_cat[cat].append((w, carrier, pred, span_ok, round(conf, 3)))
                if span_ok:
                    span_correct += 1
                    span_by_cat[cat][0] += 1

    print("=" * 70)
    print(f"词级探针结果（{len(LEXICON_BY_CAT)} 词 × {len(CARRIERS)} 载体 = {total} 次）")
    print("=" * 70)
    print(f"整体类别正确率: {correct/total*100:.1f}%  ({correct}/{total})")
    print(f"整体定位正确率: {span_correct/span_total*100:.1f}%  ({span_correct}/{span_total})")
    print()
    print(f"  {'类别':12s} {'类别准确率':>10s} {'定位准确率':>10s}")
    for cat in (1, 2, 3):
        c, t = per_cat[cat]
        sc, st = span_by_cat[cat]
        print(f"  {CAT_NAME[cat]:12s} {c/t*100:9.1f}% {sc/st*100:9.1f}%")

    print()
    print("=" * 70)
    print("判错的词（按期望类别分组）")
    print("=" * 70)
    for cat in (1, 2, 3):
        bad = wrong_by_cat[cat]
        if not bad:
            print(f"\n【{CAT_NAME[cat]}】全部正确 ✅")
            continue
        # 同一个词可能在多个载体上都错，按词聚合
        by_word = defaultdict(list)
        for w, carrier, pred, span_ok, conf in bad:
            by_word[w].append((pred, span_ok, conf))
        print(f"\n【{CAT_NAME[cat]}】{len(by_word)}/{len(LEXICON_BY_CAT[cat])} 个词在至少一个载体上判错：")
        for w, hits in sorted(by_word.items(), key=lambda x: -len(x[1])):
            preds = sorted({CAT_NAME[p] for p, _, _ in hits})
            n = len(hits)
            print(f"  {w:8s}  错 {n}/{len(CARRIERS)} 载体  -> 被判为 {', '.join(preds)}")

    print()
    print("=" * 70)
    print(f"判错词总数: {sum(len(v) for v in wrong_by_cat.values())} / {total} 次预测")


if __name__ == "__main__":
    main()
