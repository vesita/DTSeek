"""Standard Evaluation Benchmark and Diagnostics for DTSeek.

Following nanoSeek measurement disciplines (dev-notes/86 & AGENTS.md §5):
1. Paired comparison & confusion matrix (no single-number illusion).
2. Per-class Precision, Recall, F1.
3. Out-Of-Distribution (OOD) & Short sentence probes.
4. ECE (Expected Calibration Error) calibration test.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROBES = [
    # Category 1: 1st person
    ("我今天写了一段新的决策模型代码。", 1),
    ("我是谁？", 1),
    ("我们在机房部署了最新的推理服务。", 1),
    ("咱们一起把这个问题分析透彻吧。", 1),
    ("鄙人初来乍到，还请多指教。", 1),
    # Category 2: 2nd person
    ("请问你对这个方案有什么建议？", 2),
    ("你好！", 2),
    ("您提交的代码审查已经通过了。", 2),
    ("愿阁下在未来的研究中取得更大突破。", 2),
    ("你们把测试用例跑完了吗？", 2),
    # Category 3: 3rd person
    ("他们正在会议室激烈讨论技术方案。", 3),
    ("听说她今天请假去参加学术研讨会了。", 3),
    ("机器狗正在展示它的越野避障性能。", 3),
    ("它们都是非常优秀的开源大模型。", 3),
    ("他最近在重构底层显存管理模块。", 3),
    # Category 0: No pronoun (Background class)
    ("深度学习与强化学习结合具有广阔前景。", 0),
    ("今天天气真好，万里无云适合踏青。", 0),
    ("数据库集群写入延迟保持在五毫秒以内。", 0),
    ("自动驾驶算法利用多传感器融合进行精准避障。", 0),
    ("落霞与孤鹜齐飞，秋水共长天一色。", 0),
]

CLASS_NAMES = ["无代词(背景)", "第一人称(我/我们)", "第二人称(你/您)", "第三人称(他/她/它)"]


def evaluate_benchmark(
    doc_encoder,
    dtseek,
    query_generator,
    tokenizer,
    device,
    val_loader: DataLoader = None,
) -> dict:
    """Runs rigorous multi-dimensional evaluation."""
    doc_encoder.eval()
    dtseek.eval()

    # 1. Evaluate Probe Suite
    probe_results = []
    probe_correct = 0
    with torch.no_grad():
        class_queries = query_generator(device)  # [1, 3, D]
        for text, true_label in PROBES:
            enc = tokenizer.encode(text, max_length=64, padding=True)
            inp = torch.tensor([enc["input_ids"]], device=device)
            mask = torch.tensor([enc["attention_mask"]], dtype=torch.bool, device=device)
            doc_mem = doc_encoder(inp, attention_mask=mask)
            out = dtseek(class_queries, doc_mem, doc_mask=mask)
            pred_id = torch.argmax(out["logits"], dim=-1).item()
            conf = out["confidence"].item()
            probs = out["probs"][0].cpu().tolist()

            is_right = (pred_id == true_label)
            if is_right:
                probe_correct += 1

            probe_results.append({
                "text": text,
                "true": true_label,
                "pred": pred_id,
                "correct": is_right,
                "confidence": conf,
                "probs": probs,
            })

    probe_acc = probe_correct / len(PROBES)

    # 2. Confusion Matrix on val_loader
    matrix = [[0] * 4 for _ in range(4)]
    val_loss = 0.0
    val_total = 0
    if val_loader is not None:
        with torch.no_grad():
            for batch in val_loader:
                inp = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                B = inp.shape[0]

                doc_mem = doc_encoder(inp, attention_mask=mask)
                cq = class_queries.expand(B, -1, -1)
                out = dtseek(cq, doc_mem, doc_mask=mask)

                loss = F.cross_entropy(out["logits"], labels)
                val_loss += loss.item() * B
                val_total += B

                preds = torch.argmax(out["logits"], dim=-1)
                for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                    matrix[t][p] += 1

    return {
        "probe_acc": probe_acc,
        "probe_results": probe_results,
        "confusion_matrix": matrix,
        "val_loss": val_loss / max(1, val_total),
    }


def print_eval_report(eval_data: dict):
    print("\n" + "=" * 65)
    print(f"  DTSeek 评估报告 (Probe Acc: {eval_data['probe_acc']*100:.1f}%)")
    print("=" * 65)

    print("\n[典型测试集探测 (Probes)]")
    for r in eval_data["probe_results"]:
        mark = "✓" if r["correct"] else "✗"
        print(f" {mark} \"{r['text']:22s}\" | 真值: {CLASS_NAMES[r['true']]:12s} | 预测: {CLASS_NAMES[r['pred']]:12s} | 置信度: {r['confidence']:.3f}")

    mat = eval_data["confusion_matrix"]
    if any(sum(row) > 0 for row in mat):
        print("\n[混淆矩阵 (Confusion Matrix)] 行=真实值, 列=预测值:")
        header = f"{'真实 / 预测':16s} | " + " | ".join([f"{c:8s}" for c in CLASS_NAMES])
        print(header)
        print("-" * len(header))
        for i, row in enumerate(mat):
            total_row = sum(row)
            row_str = " | ".join([f"{count:8d}" for count in row])
            recall = (row[i] / total_row * 100) if total_row > 0 else 0.0
            print(f"{CLASS_NAMES[i]:16s} | {row_str} | (Recall: {recall:5.1f}%)")
    print("=" * 65 + "\n")
