"""DTSeek 多任务决策引擎演示：一次输入，多张任务卡同时出结果。

类别名、配色、发射步数**全部来自 ckpt 里的 TaskSpec 快照**，
本文件不再持有任何一份类别定义 —— 加任务不用改这里。

    uv run python examples/example_multitask.py "我不太喜欢这个方案"
    uv run python examples/example_multitask.py --tasks sentiment "开心"
"""
import argparse

from dtseek.tasks.engine import DEFAULT_CKPT, MultiTaskEngine

RESET = "\033[0m"


def show(engine: MultiTaskEngine, line: str, tasks: list[str] | None = None):
    res = engine.predict(line, tasks=tasks)
    if "error" in res:
        print(f"  {res['error']}")
        return

    print("\n" + "=" * 68)
    print(f"输入: {res['text']}   (自动分句 {res['num_segments']} 段)")
    print("=" * 68)
    for task, anchors in res["tasks"].items():
        print(f"\n【{engine.task_label(task)}】{task}  ← TaskSpec.max_steps={engine.specs[task].max_steps}")
        print(f"  高亮: {engine.render(res['text'], anchors)}")
        if not anchors:
            print("  切片: 无（背景/未触发）")
            continue
        for a in anchors:
            snip = res["text"][a["s0"]:a["e0"] + 1]
            pair = f" 第{a['pair_index']}对·{a['pair_side']}" if "pair_index" in a else ""
            print(f"  Step {a['step']}: {a['color']}[{a['s0']+1}:{a['e0']+1}]{RESET} "
                  f"{a['category']:12s} 置信度={a['confidence']:.3f} "
                  f"'{snip}' {a['next_action']}{pair}")
    print()


def main(argv: list[str] | None = None, default_tasks: list[str] | None = None):
    ap = argparse.ArgumentParser(description="DTSeek 多任务决策引擎演示")
    ap.add_argument("text", nargs="?", help="待分析文本；留空进入交互模式")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tasks", nargs="*", default=default_tasks,
                    help="只跑这几张任务卡；留空 = ckpt 里的全部")
    args = ap.parse_args(argv)

    engine = MultiTaskEngine(ckpt_path=args.ckpt)
    if args.tasks:
        missing = [t for t in args.tasks if t not in engine.decoders]
        if missing:
            ap.error(f"ckpt 里没有任务卡 {missing}；可用：{engine.tasks}")

    if args.text:
        show(engine, args.text, args.tasks)
        return

    labels = " / ".join(engine.task_label(t) for t in (args.tasks or engine.tasks))
    print("\n" + "=" * 68)
    print("  DTSeek 多任务决策引擎（共享基座 + 可插拔任务卡）")
    print(f"  一次输入，同时给出：{labels}")
    print("  输入 exit 退出")
    print("=" * 68)
    while True:
        try:
            line = input("\nDTSeek > ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                break
            show(engine, line, args.tasks)
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()
