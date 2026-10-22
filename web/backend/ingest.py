"""Load the result CSVs in results/ into web/homebot.db (idempotent: experiments and runs are rebuilt).

python -m web.backend.ingest
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys

from .db import ROOT, RUN_COLUMNS, connect

RESULTS = ROOT / "results"

# CSV column -> common run column
RENAME = {"person_contacts_moving": "person_contacts_robot_moving", "perc_err_mm": "perception_err_mm"}

EXPERIMENTS = {
    "E1-L1": ("E1 · Mức 1", "Điều hướng tới bàn: nhà trống, đồ đạc xê dịch (lớp an toàn bật)."),
    "E1-L2": ("E1 · Mức 2", "Điều hướng với một người đi lại trong nhà (lớp an toàn bật)."),
    "E1-L3": ("E1 · Mức 3", "Điều hướng với người và thú cưng (lớp an toàn bật)."),
    "E4": ("E4 · Tắt an toàn", "Như E1 mức 2 nhưng tắt lớp an toàn: đo giá trị của lớp an toàn."),
    "E2-pick": ("E2 · Nhặt lon", "Kỹ năng thao tác: nhặt lon trên bàn từ vị trí đứng bị nhiễu."),
    "E2-place": ("E2 · Đặt vào bồn", "Kỹ năng thao tác: đặt lon vào bồn rửa."),
    "E2-place-sink-moved": ("E2 · Đặt, bồn dời chỗ", "Đặt lon khi bồn rửa bị dời dọc mặt bếp (robot tự tìm bồn bằng camera)."),
    "E3": ("E3 · Toàn bộ nhiệm vụ", "\"Mang lon tới bồn rửa\" từ phòng khách, mức 2 (có người đi lại)."),
    "E3-sink-moved": ("E3 · Bồn dời chỗ", "Toàn bộ nhiệm vụ khi bồn rửa bị dời dọc mặt bếp."),
    "E5": ("E5 · Người cao tuổi", "Toàn bộ nhiệm vụ khi người sống cùng là người cao tuổi (đi chậm, dừng, do dự)."),
}


def parse(v: str):
    if v is None:
        return None
    s = v.strip()
    if s == "":
        return None
    if s in ("True", "False"):
        return s == "True"
    try:
        x = float(s)
    except ValueError:
        if s[:1] in "[{":
            try:
                return json.loads(s)
            except ValueError:
                pass
        return s
    if math.isnan(x) or math.isinf(x):
        return None
    return int(x) if re.fullmatch(r"-?\d+", s) else x


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_run(experiment: str, raw: dict, policy_default="expert") -> dict:
    row = {RENAME.get(k, k): parse(v) for k, v in raw.items()}
    run = {c: row.pop(c, None) for c in RUN_COLUMNS}
    run["experiment"] = experiment
    run["policy"] = run["policy"] or policy_default
    for k in ("success", "fell"):
        if run[k] is not None:
            run[k] = int(bool(run[k]))
    run["extra"] = json.dumps(row, ensure_ascii=False)
    return run


def collect():
    """(experiment id, source file name, runs) for every result file present."""
    out = []
    nav = RESULTS / "e1_e4_nav.csv"
    if nav.exists():
        for raw in read_csv(nav):
            exp = f"E1-L{raw['level']}" if raw["safety"] == "True" else ("E4" if raw["level"] == "2" else
                                                                         f"E4-L{raw['level']}")
            out.append((exp, nav.name, to_run(exp, raw)))
    for p in sorted(RESULTS.glob("e2_manip_*.csv")):
        for raw in read_csv(p):
            exp = f"E2-{raw.get('skill', 'pick')}"
            out.append((exp, p.name, to_run(exp, raw, p.stem.removeprefix("e2_manip_"))))
    p = RESULTS / "e2_place_sink_moved.csv"
    if p.exists():
        for raw in read_csv(p):
            out.append(("E2-place-sink-moved", p.name, to_run("E2-place-sink-moved", raw)))
    for p in sorted(RESULTS.glob("e3_full_level2_*.csv")):
        exp = "E3-sink-moved" if p.stem.endswith("sink_moved") else "E3"
        for raw in read_csv(p):
            out.append((exp, p.name, to_run(exp, raw, p.stem.removeprefix("e3_full_level2_"))))
    p = RESULTS / "e5_elderly.csv"
    if p.exists():
        for raw in read_csv(p):
            out.append(("E5", p.name, to_run("E5", raw)))
    return out


def ingest(con=None) -> dict:
    con = con or connect()
    rows = collect()
    counts: dict[str, int] = {}
    sources: dict[str, set] = {}
    for exp, src, _ in rows:
        counts[exp] = counts.get(exp, 0) + 1
        sources.setdefault(exp, set()).add(src)
    with con:
        con.execute("DELETE FROM runs")
        con.execute("DELETE FROM sqlite_sequence WHERE name = 'runs'")     # stable run ids across re-ingests
        con.execute("DELETE FROM experiments")
        order = list(EXPERIMENTS)
        for exp in counts:
            name, desc = EXPERIMENTS.get(exp, (exp, ""))
            con.execute("INSERT INTO experiments(id, name, description, source, sort) VALUES (?,?,?,?,?)",
                        (exp, name, desc, ", ".join(sorted(sources[exp])),
                         order.index(exp) if exp in order else len(order)))
        cols = RUN_COLUMNS + ["extra"]
        con.executemany(f"INSERT INTO runs({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                        [[r[c] for c in cols] for _, _, r in rows])
    return counts


def main():
    counts = ingest()
    per_file: dict[str, int] = {}
    for _, src, _ in collect():
        per_file[src] = per_file.get(src, 0) + 1
    for exp, n in counts.items():
        print(f"{exp:22s} {n:4d} runs")
    print("per file:", json.dumps(per_file))
    print("total", sum(counts.values()))


if __name__ == "__main__":
    sys.exit(main())
