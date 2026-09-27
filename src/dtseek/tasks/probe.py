"""任务无关的词级探针。

回答的问题：**有没有整项没学会**。
句级验证集准确率会掩盖这件事 —— 高频词样本多、低频词样本少，平均下来看不出
某个词从来没被学会（情绪任务实测：句级 98%，词级只有 43.7%）。

做法：把任务卡声明的每个探针单元（词 / 词对）填进多种载体句，逐个测类别与定位。

两条纪律：
  1. **先过已知答案对照** —— 探针自己在几句话上必须先给出预期结论，否则先怀疑探针；
  2. **载体必须覆盖无终止标点的裸词形态** —— 只测"…。"会漏掉定位被截断的缺陷。

probe 逻辑只认 `ProbeUnit`，不认识任何具体任务；任务卡自己声明考什么。
"""
from __future__ import annotations

from collections import defaultdict

from dtseek.tasks.engine import MultiTaskEngine
from dtseek.tasks.plugin import ProbeUnit, probe_units_of


def build_cases(units: list[ProbeUnit]) -> list[dict]:
    """把探针单元展开成待测样本，并**在生成时**算好每个词在句子里的期望区间。

    fail-closed：载体模板本身若已含该词，期望区间会算错 —— 直接抛错而不是静默测错。
    """
    cases = []
    for unit in units:
        for carrier, text in zip(unit.carriers, unit.sentences()):
            for w in unit.words:
                if w in carrier:
                    raise ValueError(
                        f"探针载体 {carrier!r} 本身含有待测词 {w!r}（单元 {unit.key!r}），"
                        " 期望区间会算错；请换一个不含该词的载体模板")
            spans = []
            cursor = 0
            for w in unit.words:
                idx = text.find(w, cursor)
                if idx < 0:
                    raise ValueError(f"载体 {carrier!r} 展开后找不到词 {w!r}：{text!r}")
                spans.append((w, idx, idx + len(w) - 1))
                cursor = idx + len(w)
            cases.append({
                "key": unit.key,
                "expected_class": unit.expected_class,
                "carrier": carrier,
                "text": text,
                "expected_spans": spans,
            })
    return cases


def sanity_check(engine: MultiTaskEngine, task: str) -> dict:
    """已知答案对照：给几句话，探针必须给出预期类别，否则先怀疑探针。"""
    spec = engine.specs[task]
    cases = getattr(engine_card(task), "sanity_cases", lambda: [])()
    results = []
    for text, want in cases:
        anchors = engine.predict(text, tasks=[task])["tasks"][task]
        got = anchors[0]["class_id"] if anchors else 0
        results.append({"text": text, "want": want, "got": got, "ok": got == want,
                        "want_name": spec.classes[want].display,
                        "got_name": spec.classes[got].display})
    return {"passed": all(r["ok"] for r in results), "cases": results}


def engine_card(task: str):
    from dtseek.tasks.plugin import all_tasks
    return all_tasks()[task]


def run_probe(engine: MultiTaskEngine, task: str,
              units: list[ProbeUnit] | None = None) -> dict:
    """跑一个任务的词级探针，返回可判对错的统计。

    单切片任务只看**首个切片**（和历史上的探针口径一致，便于纵向比较）；
    成对任务看完整发射序列，按对评估。
    """
    spec = engine.specs[task]
    units = units if units is not None else probe_units_of(engine_card(task))
    cases = build_cases(units)

    per_class: dict[int, list[int]] = defaultdict(lambda: [0, 0])   # cat -> [ok, total]
    per_key: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # key -> [类别对, 定位对, 总]
    failures = []
    pair_exact = pair_total = 0

    for case in cases:
        anchors = engine.predict(case["text"], tasks=[task])["tasks"][task]
        want = case["expected_class"]
        expected_spans = case["expected_spans"]

        if spec.pair_emission:
            # 类别与定位**分开算**：只报一个合起来的数，看不出到底是「认成了另一种关系」
            # 还是「词根本没找着」—— 这两种失败的修法完全不同。
            cls_ok = [a["class_id"] for a in anchors] == [want] * len(expected_spans)
            loc_ok = (len(anchors) == len(expected_spans)
                      and all(a["s0"] == s and a["e0"] == e
                              for a, (_, s, e) in zip(anchors, expected_spans)))
            pair_total += 1
            pair_exact += (cls_ok and loc_ok)
        elif not anchors:
            cls_ok = loc_ok = False
        else:
            _, s0, e0 = expected_spans[0]
            first = anchors[0]
            cls_ok = first["class_id"] == want
            loc_ok = first["s0"] == s0 and first["e0"] == e0
        span_ok = loc_ok

        per_class[want][1] += 1
        per_class[want][0] += cls_ok
        per_key[case["key"]][0] += cls_ok
        per_key[case["key"]][1] += span_ok
        per_key[case["key"]][2] += 1
        if not (cls_ok and span_ok):
            failures.append({
                "key": case["key"], "carrier": case["carrier"], "text": case["text"],
                "want": want, "want_name": spec.classes[want].display,
                "got_names": [a["category"] for a in anchors] or ["（未触发）"],
                "expected": [f"{w}[{s}:{e}]" for w, s, e in expected_spans],
                "got": [f"{a['category']}[{a['s0']}:{a['e0']}]" for a in anchors],
                "ok_class": cls_ok, "ok_span": span_ok,
            })

    n = len(cases)
    report = {
        "task": task,
        "n_units": len(units),
        "n_cases": n,
        "class_acc": sum(v[0] for v in per_class.values()) / max(1, n),
        "span_acc": (n - sum(1 for f in failures if not f["ok_span"])) / max(1, n),
        "by_class": {c: {"acc": v[0] / max(1, v[1]), "ok": v[0], "total": v[1]}
                     for c, v in sorted(per_class.items())},
        "by_key": {k: {"cls_ok": v[0], "span_ok": v[1], "total": v[2],
                       "cls_acc": v[0] / max(1, v[2]), "span_acc": v[1] / max(1, v[2])}
                   for k, v in sorted(per_key.items())},
        "partial_keys": sorted(k for k, v in per_key.items() if v[0] < v[2] or v[1] < v[2]),
        "failures": failures,
    }
    if spec.pair_emission:
        report["pair_exact"] = pair_exact / max(1, pair_total)
        report["n_pairs"] = pair_total
    return report
