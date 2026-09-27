"""从开源词典抽出成语识别任务的句子语料，生成 `src/dtseek/tasks/builtin/idiom/examples.py`。

为什么要单独抽一份语料，而不是在训练时现场挖对话语料：
  - 对话语料里成语极稀疏（实测 20 万行只挖到 326 句），且**正面样本是文学语体、
    背景样本是口语**，模型会学成「口语 ⇒ 没有成语」这种语体捷径；
  - 词典自带的 `example.text` 用 `～` 占位成语，替换后就是**真实例句**，
    能覆盖 2214 个白名单成语里的 2134 个。

所以这里同时抽出两类、且**语体对齐**：
  - `EXAMPLES`   ：词典例句（替换 `～`），正面样本
  - `BACKGROUND` ：词典的出处引文与释义文本里**不含任何白名单成语**的句子，背景样本

用法：uv run python scripts/build_idiom_corpus.py
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks.builtin.idiom.lexicon import IDIOMS  # noqa: E402

MAX_LEN = 64
MIN_LEN = 4
MAX_BACKGROUND = 8000


def load_jsonlish(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return [json.loads(l.strip().rstrip(",")) for l in raw.splitlines()
                if l.strip() not in ("", "[", "]")]


def has_idiom(text: str) -> bool:
    return any(text[i:i + 4] in IDIOMS for i in range(len(text) - 3))


def main() -> int:
    ap = argparse.ArgumentParser(description="抽出成语识别的例句与背景句")
    ap.add_argument("--dict-dir", type=Path, default=Path("/tmp/chinese-dictionary"))
    ap.add_argument("--out", type=Path, default=ROOT / "src" / "dtseek" / "tasks" / "builtin" / "idiom" / "examples.py")
    args = ap.parse_args()

    rows = load_jsonlish(args.dict_dir / "idiom" / "idiom.json")
    print(f"词典条目 {len(rows)}")

    examples: dict[str, str] = {}       # 句子 -> 目标成语（去重）
    for r in rows:
        w = r.get("word")
        ex = r.get("example")
        if w not in IDIOMS or not isinstance(ex, dict) or not ex.get("text"):
            continue
        text = ex["text"].replace("～", w).strip()
        if MIN_LEN <= len(text) <= MAX_LEN and w in text:
            examples.setdefault(text, w)
    print(f"例句（替换 ～ 后含目标成语）：{len(examples)} 条，"
          f"覆盖 {len({w for w in examples.values()})}/{len(IDIOMS)} 个成语")

    background: list[str] = []
    seen_bg = set()
    for r in rows:
        for field in ("source", "explanation"):
            val = r.get(field)
            text = val.get("text") if isinstance(val, dict) else val
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not (MIN_LEN <= len(text) <= MAX_LEN):
                continue
            if text in seen_bg or has_idiom(text):
                continue
            seen_bg.add(text)
            background.append(text)
            if len(background) >= MAX_BACKGROUND:
                break
        if len(background) >= MAX_BACKGROUND:
            break
    print(f"背景句（不含任何白名单成语）：{len(background)} 条")

    if len(background) < 2000:
        print("⚠ 背景句偏少，模型可能学成永远开火")

    lines = [
        '"""成语识别语料（由 scripts/build_idiom_corpus.py 从开源词典生成）。',
        "",
        "来源：https://github.com/mapull/chinese-dictionary (MIT)，commit e804ada",
        "不要手改本文件；改生成脚本后重跑。",
        '"""',
        "from __future__ import annotations",
        "",
        f"#: 例句：(目标成语, 句子)。词典 example 用 ～ 占位，已替换成成语。共 {len(examples)} 条",
        "EXAMPLES: tuple[tuple[str, str], ...] = (",
    ]
    for text, w in sorted(examples.items()):
        lines.append(f"    ({w!r}, {text!r}),")
    lines.append(")")
    lines.append("")
    lines.append(f"#: 背景句：词典出处引文与释义里不含任何白名单成语的句子。共 {len(background)} 条")
    lines.append("BACKGROUND: tuple[str, ...] = (")
    for text in background:
        lines.append(f"    {text!r},")
    lines.append(")")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写出 {args.out}（{args.out.stat().st_size / 1024:.0f} KB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
