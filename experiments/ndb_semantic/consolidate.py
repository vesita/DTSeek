#!/usr/bin/env python
"""把全部原始产物合成一份 `results.json`（交付用）。

输入：
  q1_kind_decomposition.json / oracle_bound.json / oracle_separability.json /
  oracle_metric.json / oracle_vq_learned.json / oracle_codebook.json /
  by_kind_ref.jsonl（既有 base/literal 卡的按类别拆解） / ab_run.log（semantic 三臂）
输出：results.json
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
REF = Path("/tmp/ab_ndb2")
KEYS = ("repeat_mention_acc", "first_mention_acc", "id_acc", "cluster_f1", "bg_fp",
        "exact_match", "span_hit")


def m_of(p: Path):
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("AB_METRICS "):
            return json.loads(line[len("AB_METRICS "):])
    return None


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    rows: dict[tuple[str, int], dict] = {}

    # 1) 既有参考卡的按类别拆解（eval_by_kind.py 产出，口径已与 evaluate_task 对齐）
    bk_file = OUT / "by_kind_ref.jsonl"
    if bk_file.exists():
        for line in bk_file.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            arm = "base" if r["tag"].startswith("person_base") else "literal"
            rows.setdefault((arm, r["seed"]), {})["by_kind"] = r["by_kind"]

    # 2) 参考卡的主指标（原始训练日志）
    for seed in (42, 43):
        for name, arm in (("base", "base"), ("ndb", "literal")):
            r = m_of(REF / f"person_{name}_seed{seed}.log")
            if r:
                rows.setdefault((arm, seed), {}).update(r)

    # 3) 本次 AB（semantic + 复刻校验臂）
    ab_log = OUT / "ab_run.log"
    if ab_log.exists():
        for line in ab_log.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.startswith("AB_METRICS "):
                continue
            r = json.loads(line[len("AB_METRICS "):])
            tag = r.get("tag", "")
            if tag.startswith("semantic_"):
                rows.setdefault(("semantic", r["seed"]), {}).update(r)
            elif tag.startswith("validate_"):
                rows.setdefault((f"CHECK-{r['arm']}", r["seed"]), {}).update(r)

    res = {k: load(OUT / f) for k, f in (
        ("q1_kind_decomposition", "q1_kind_decomposition.json"),
        ("q2_oracle_raw", "oracle_bound.json"),
        ("q2_separability", "oracle_separability.json"),
        ("q2_learned_metric", "oracle_metric.json"),
        ("q2_vq_learned", "oracle_vq_learned.json"),
        ("q2_codebook_choice", "oracle_codebook.json"))}

    res["ab"] = {}
    for (arm, seed), r in sorted(rows.items()):
        if "metrics" not in r:
            continue
        res["ab"][f"{arm}|seed{seed}"] = {
            "metrics": {k: r["metrics"].get(k) for k in KEYS},
            "by_kind": r.get("by_kind"),
            "cost": {"sec_per_step": r["sec_per_step"], "peak_mem_mb": r["peak_mem_mb"],
                     "n_head_params": r["n_head_params"], "n_ndb_params": r["n_ndb_params"],
                     "n_frozen_stats": r.get("n_frozen_stats", 0),
                     "ndb_table_gb": r.get("ndb_table_gb", 0.0)},
        }

    def paired(a, b):
        got = {}
        for k in KEYS:
            ds = [rows[(a, s)]["metrics"][k] - rows[(b, s)]["metrics"][k] for s in (42, 43)]
            got[k] = {"deltas": ds, "mean": statistics.fmean(ds),
                      "range": max(ds) - min(ds)}
        return got

    res["paired_delta_vs_base"] = {a: paired(a, "base") for a in ("literal", "semantic")}
    res["paired_delta_semantic_minus_literal"] = paired("semantic", "literal")
    res["paired_delta_semantic_minus_literal_by_kind"] = {
        k: {"semantic": [rows[("semantic", s)]["by_kind"][k]["acc"] for s in (42, 43)],
            "literal": [rows[("literal", s)]["by_kind"][k]["acc"] for s in (42, 43)]}
        for k in ("literal_same_id", "alias", "literal_other_id")}

    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print("写出", OUT / "results.json")
    print(json.dumps(res["paired_delta_semantic_minus_literal"], ensure_ascii=False, indent=2))
    print(json.dumps(res["paired_delta_semantic_minus_literal_by_kind"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
