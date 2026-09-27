"""通用词级探针 CLI：对任意任务卡跑「有没有整项没学会」的检查。

    uv run python scripts/eval_probe.py --task sentiment
    uv run python scripts/eval_probe.py --task relation --limit 50
    uv run python scripts/eval_probe.py                 # 跑 ckpt 里全部任务卡

先跑已知答案对照；对照不过会明确提示「先怀疑探针」。
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dtseek.tasks.probe import run_probe, sanity_check  # noqa: E402
from dtseek.tasks.engine import DEFAULT_CKPT, MultiTaskEngine  # noqa: E402
from dtseek.tasks.plugin import probe_units_of  # noqa: E402


def report(engine: MultiTaskEngine, task: str, limit: int, show_failures: int) -> bool:
    spec = engine.specs[task]
    from dtseek.tasks.plugin import all_tasks
    units = probe_units_of(all_tasks()[task])
    if limit:
        units = units[:limit]
    if not units:
        print(f"  [{task}] 该任务卡没有声明词级探针单元，跳过")
        return True

    print("=" * 74)
    print(f"【{spec.label}】{task}  词级探针")
    print("=" * 74)

    sane = sanity_check(engine, task)
    if sane["cases"]:
        for c in sane["cases"]:
            print(f"  {'✓' if c['ok'] else '✗'} 已知答案 '{c['text']}' "
                  f"期望={c['want_name']} 实际={c['got_name']}")
        if not sane["passed"]:
            print("\n⚠ 已知答案对照未通过：**先怀疑探针写错**，再怀疑模型。\n")
        else:
            print("  对照全部通过 ✅ 探针可信\n")

    res = run_probe(engine, task, units)
    print(f"  {res['n_units']} 个单元 × 载体 = {res['n_cases']} 次预测")
    print(f"  类别正确率: {res['class_acc']*100:5.1f}%")
    print(f"  定位正确率: {res['span_acc']*100:5.1f}%")
    if "pair_exact" in res:
        print(f"  对完整命中率: {res['pair_exact']*100:5.1f}%  ({res['n_pairs']} 对)")
    for cat, agg in res["by_class"].items():
        print(f"    {spec.classes[cat].display:12s} {agg['acc']*100:5.1f}%  ({agg['ok']}/{agg['total']})")

    if res["partial_keys"]:
        print(f"\n  至少一项判错的单元 {len(res['partial_keys'])} 个：")
        by_key = defaultdict(list)
        for f in res["failures"]:
            by_key[f["key"]].append(f)
        for key in sorted(by_key, key=lambda k: -len(by_key[k])):
            hits = by_key[key]
            got = sorted({g for h in hits for g in h["got_names"]})
            agg = res["by_key"][key]
            n = agg["total"]
            print(f"    {key:10s} 类别错 {n - agg['cls_ok']}/{n}  定位错 {n - agg['span_ok']}/{n}"
                  f"  -> 被判为 {', '.join(got)}")
        if show_failures:
            print("\n  逐条失败样例：")
            for f in res["failures"][:show_failures]:
                print(f"    ✗ '{f['text']}'  期望={f['want_name']}{f['expected']}  "
                      f"实际={'|'.join(f['got'])}")
    else:
        print("\n  全部单元在所有载体上都对 ✅")
    print()
    return sane["passed"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DTSeek 通用词级探针")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--task", default=None, help="只跑这张任务卡；留空 = ckpt 里全部")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个探针单元（调试用）")
    ap.add_argument("--show-failures", type=int, default=0, help="打印前 N 条失败样例")
    args = ap.parse_args(argv)

    engine = MultiTaskEngine(ckpt_path=args.ckpt)
    tasks = [args.task] if args.task else engine.tasks
    missing = [t for t in tasks if t not in engine.decoders]
    if missing:
        ap.error(f"ckpt 里没有任务卡 {missing}；可用：{engine.tasks}")

    all_sane = True
    for task in tasks:
        all_sane &= report(engine, task, args.limit, args.show_failures)
    return 0 if all_sane else 1


if __name__ == "__main__":
    sys.exit(main())
