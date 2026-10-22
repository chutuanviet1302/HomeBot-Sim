"""HomeBot-Sim results API + site.

uvicorn web.backend.app:app --port 8000      (from the repository root)
"""
from __future__ import annotations

import mimetypes
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import replay as rp
from .db import ROOT, connect, run_row
from .ingest import ingest

JOB_DIR = Path(__file__).resolve().parent / "jobs"
DIST = ROOT / "web" / "frontend" / "dist"
REPORT = ROOT / "report"
EXPECTED_SIM_S = {1: 125.0, 2: 200.0, 3: 210.0}     # typical episode length (s simulated), for the progress bar

con = connect()
if not con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]:
    ingest(con)
with con:   # jobs that were running when the server stopped
    con.execute("UPDATE jobs SET status = 'failed', result = 'interrupted' WHERE status IN ('running', 'queued')")
rp.sync(con)

app = FastAPI(title="HomeBot-Sim")
lock = threading.Lock()


# ---------------------------------------------------------------- helpers
def runs(exp: str, policy: str | None = None) -> list[dict]:
    q, args = "SELECT * FROM runs WHERE experiment = ?", [exp]
    if policy:
        q += " AND policy = ?"
        args.append(policy)
    with lock:
        return [run_row(r) for r in con.execute(q, args).fetchall()]


def rate(rs):
    return sum(1 for r in rs if r["success"]) / len(rs) if rs else None


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def total(rs, k):
    return sum(r[k] or 0 for r in rs)


def summary(rs) -> dict:
    ok = [r for r in rs if r["success"]]
    return {"n": len(rs), "success": len(ok), "success_rate": rate(rs),
            "person_contacts": total(rs, "person_contacts"),
            "person_contacts_robot_moving": total(rs, "person_contacts_robot_moving"),
            "runs_with_contact": sum(1 for r in rs if (r["person_contacts_robot_moving"] or 0) > 0),
            "min_person_dist_median": med(r["min_person_dist"] for r in rs),
            "min_person_dist_min": min((r["min_person_dist"] for r in rs if r["min_person_dist"] is not None),
                                       default=None),
            "arm_stops": total(rs, "arm_stops"), "arm_contacts": total(rs, "arm_contacts"),
            "furniture_contacts": total(rs, "furniture_contacts"), "falls": total(rs, "fell"),
            "time_s_mean_success": mean(r["time_s"] for r in ok),
            "basin_err_mm_mean": mean(r["basin_err_mm"] for r in rs),
            "perception_err_mm_mean": mean(r["perception_err_mm"] for r in rs),
            "failed_at": {k: sum(1 for r in rs if r["failed_at"] == k)
                          for k in sorted({r["failed_at"] for r in rs if r["failed_at"]})}}


def policies_of(exp: str) -> list[str]:
    with lock:
        return [r[0] for r in con.execute("SELECT DISTINCT policy FROM runs WHERE experiment = ? ORDER BY policy",
                                          (exp,)).fetchall()]


def pct(x):
    return None if x is None else round(100 * x, 1)


# ---------------------------------------------------------------- results
@app.get("/api/kpi")
def kpi():
    cards = []
    e3, e3s = runs("E3", "expert"), runs("E3-sink-moved", "expert")
    if e3:
        s = summary(e3)
        cards.append({"id": "e3_success", "label": "Hoàn thành nhiệm vụ (E3)", "value": f"{s['success']}/{s['n']}",
                      "pct": pct(s["success_rate"]), "sub": "\"Mang lon tới bồn rửa\", có người đi lại",
                      "tone": "good"})
    if e3s:
        s = summary(e3s)
        cards.append({"id": "e3_sink", "label": "Bồn rửa bị dời chỗ", "value": f"{s['success']}/{s['n']}",
                      "pct": pct(s["success_rate"]), "sub": "Không lập trình lại: robot tự tìm bồn bằng camera",
                      "tone": "good"})
    safe, unsafe = runs("E1-L2"), runs("E4")
    safety = None
    if safe and unsafe:
        a, b = summary(safe), summary(unsafe)
        safety = {"with": a, "without": b,
                  "min_dist_with": sorted(r["min_person_dist"] for r in safe if r["min_person_dist"] is not None),
                  "min_dist_without": sorted(r["min_person_dist"] for r in unsafe
                                             if r["min_person_dist"] is not None)}
        cards.append({"id": "contacts", "label": "Va chạm người khi robot di chuyển",
                      "value": f"{a['person_contacts_robot_moving']}", "compare": f"{b['person_contacts_robot_moving']}",
                      "sub": f"Có lớp an toàn vs. tắt (E4), mỗi bên {a['n']} lượt", "tone": "good"})
        cards.append({"id": "min_dist", "label": "Khoảng cách tối thiểu tới người (trung vị)",
                      "value": f"{a['min_person_dist_median']:.2f} m" if a["min_person_dist_median"] is not None else "–",
                      "compare": f"{b['min_person_dist_median']:.2f} m" if b["min_person_dist_median"] is not None else None,
                      "sub": "Giữa hai mép thân; có an toàn vs. tắt", "tone": "good"})
    full = e3 + e3s
    if full:
        s = summary(full)
        cards.append({"id": "arm_stops", "label": "Lần tay dừng vì an toàn", "value": str(s["arm_stops"]),
                      "sub": f"Trong {s['n']} lượt E3; va chạm cánh tay: {s['arm_contacts']}", "tone": "neutral"})
    with lock:
        n_all, falls = con.execute("SELECT COUNT(*), COALESCE(SUM(fell), 0) FROM runs").fetchone()
    cards.append({"id": "falls", "label": "Robot ngã", "value": str(falls), "sub": f"Trên tổng {n_all} lượt chạy",
                  "tone": "good" if falls == 0 else "bad"})
    basin = mean([r["basin_err_mm"] for r in full + runs("E2-place") + runs("E2-place-sink-moved")])
    perc = mean([r["perception_err_mm"] for r in runs("E2-pick")])
    if basin is not None or perc is not None:
        cards.append({"id": "perception", "label": "Sai số nhận thức (camera)",
                      "value": f"{basin:.1f} mm" if basin is not None else "–",
                      "compare": f"{perc:.1f} mm" if perc is not None else None,
                      "sub": "Vị trí bồn rửa · vị trí lon (trung bình)", "tone": "neutral"})

    # expert vs learned policies (ACT, ...) wherever more than one policy was evaluated
    pol = []
    for exp in ("E2-pick", "E2-place", "E2-place-sink-moved", "E3", "E3-sink-moved", "E5"):
        for p in policies_of(exp):
            s = summary(runs(exp, p))
            pol.append({"experiment": exp, "policy": p, **s})
    has_learned = any(p["policy"] != "expert" for p in pol)
    for p in pol:
        if p["policy"].startswith("act") and p["experiment"] in ("E3", "E2-pick"):
            cards.append({"id": f"act_{p['experiment']}", "label": f"Chính sách học ACT ({p['experiment']})",
                          "value": f"{p['success']}/{p['n']}", "pct": pct(p["success_rate"]),
                          "sub": p["policy"], "tone": "neutral"})
            break

    e5 = None
    e5_rows = runs("E5")
    if e5_rows:
        groups: dict[str, list] = {}
        for r in e5_rows:
            label = r["policy"]
            for k in ("person", "safety", "condition", "variant"):
                if k in r["extra"]:
                    label += f" · {k}={r['extra'][k]}"
            groups.setdefault(label, []).append(r)
        e5 = {"groups": [{"label": k, **summary(v)} for k, v in groups.items()], "reference_e3": summary(e3) if e3 else None}
        s = summary(e5_rows)
        cards.append({"id": "e5", "label": "Với người cao tuổi (E5)", "value": f"{s['success']}/{s['n']}",
                      "pct": pct(s["success_rate"]),
                      "sub": f"Va chạm khi di chuyển: {s['person_contacts_robot_moving']}", "tone": "good"})

    levels = [{"experiment": e, **summary(runs(e))} for e in ("E1-L1", "E1-L2", "E1-L3", "E4") if runs(e)]
    return {"cards": cards, "safety": safety, "policies": pol, "has_learned": has_learned, "e5": e5,
            "levels": levels, "updated": time.time()}


@app.get("/api/experiments")
def experiments():
    with lock:
        rows = con.execute("SELECT * FROM experiments ORDER BY sort").fetchall()
    return [{**dict(r), **{k: v for k, v in summary(runs(r["id"])).items() if k != "failed_at"},
             "policies": policies_of(r["id"])} for r in rows]


@app.get("/api/runs")
def list_runs(exp: str | None = None, policy: str | None = None, success: str | None = None):
    q, args = "SELECT * FROM runs WHERE 1 = 1", []
    if exp:
        q += " AND experiment = ?"
        args.append(exp)
    if policy:
        q += " AND policy = ?"
        args.append(policy)
    if success is not None and success != "":
        q += " AND success = ?"
        args.append(1 if success.lower() in ("1", "true", "yes") else 0)
    with lock:
        return [run_row(r) for r in con.execute(q + " ORDER BY id", args).fetchall()]


@app.get("/api/runs/{run_id}")
def get_run(run_id: int):
    with lock:
        r = con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if r is None:
        raise HTTPException(404, "run not found")
    return run_row(r)


@app.get("/api/replays")
def replays():
    with lock:
        return rp.sync(con)


@app.get("/api/replays/{level}/{seed}")
def get_replay(level: int, seed: int, variant: str = ""):
    data = rp.load(rp.replay_key(level, seed, variant))
    if data is None:
        with lock:
            rp.sync(con)
        data = rp.load(rp.replay_key(level, seed, variant))
    if data is None:
        raise HTTPException(404, "no recording for this level / seed")
    return JSONResponse(data)


MEDIA_FILES = ["demo.gif", "demo_full_task.mp4"]


@app.get("/api/media")
def media_list():
    """Media files available under /media (report/demo.gif, report/demo_full_task.mp4, report/figures/*)."""
    files = [f for f in MEDIA_FILES if (REPORT / f).is_file()]
    figs = REPORT / "figures"
    return files + (sorted(f"figures/{p.name}" for p in figs.iterdir() if p.is_file()) if figs.exists() else [])


# ---------------------------------------------------------------- simulation jobs
class JobIn(BaseModel):
    level: int = Field(2, ge=1, le=3)
    seed: int = Field(0, ge=0, le=99999)
    sink_range: float = Field(0.0, ge=0.0, le=0.3)
    person: str = Field("adult", pattern="^(adult|elderly)$")
    policy: str = Field("expert", pattern=r"^(expert|(act|torch):[\w./\-]+)$")


def available_policies() -> list[str]:
    pts = sorted({p.relative_to(ROOT).as_posix() for d in (ROOT, ROOT / "logs", ROOT / "checkpoints")
                  if d.exists() for p in d.glob("*.pt")})
    return ["expert"] + [f"act:{p}" for p in pts]


@app.get("/api/policies")
def policies():
    return available_policies()


def job_progress(job: dict) -> dict:
    log = Path(job["log_path"]) if job["log_path"] else None
    text = log.read_text(errors="replace") if log and log.exists() else ""
    sims = re.findall(r"simulated\s+([\d.]+) s", text)
    sim_t = float(sims[-1]) if sims else 0.0
    if job["status"] == "done":
        p = 1.0
    elif "SAVED" in text:
        p = 0.98
    else:
        p = min(0.95, sim_t / EXPECTED_SIM_S.get(job["level"], 200.0))
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return {**job, "progress": round(p, 3), "sim_t": sim_t, "log_tail": lines[-8:]}


def worker():
    global worker_busy
    while True:
        with lock:
            row = con.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                worker_busy = False
                return
            job = dict(row)
            JOB_DIR.mkdir(exist_ok=True)
            log = JOB_DIR / f"job_{job['id']}.log"
            with con:
                con.execute("UPDATE jobs SET status = 'running', started = ?, log_path = ? WHERE id = ?",
                            (time.time(), str(log), job["id"]))
        cmd = [sys.executable, "-u", str(ROOT / "web" / "backend" / "run_episode.py"), "--level", str(job["level"]),
               "--seed", str(job["seed"]), "--sink-range", str(job["sink_range"]), "--person", job["person"],
               "--policy", job["policy"]]
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        text = log.read_text(errors="replace")
        saved = re.search(r"SAVED (\S+\.npz)", text)
        res = re.search(r"RESULT (.+)", text)
        key = None
        if rc == 0 and saved:
            parsed = rp.parse_name(Path(saved.group(1)))
            key = rp.replay_key(*parsed)
            with lock:
                rp.sync(con, keys=[key])
        with lock, con:
            con.execute("UPDATE jobs SET status = ?, finished = ?, replay_key = ?, result = ? WHERE id = ?",
                        ("done" if key else "failed", time.time(), key,
                         res.group(1).strip() if res else f"exit code {rc}", job["id"]))


worker_busy = False


@app.post("/api/jobs")
def create_job(j: JobIn):
    global worker_busy
    with lock, con:
        cur = con.execute("INSERT INTO jobs(level, seed, sink_range, person, policy, status, created) "
                          "VALUES (?,?,?,?,?,'queued',?)", (j.level, j.seed, j.sink_range, j.person, j.policy,
                                                           time.time()))
        job_id = cur.lastrowid
        start = not worker_busy                                     # one simulation at a time
        worker_busy = True
    if start:
        threading.Thread(target=worker, daemon=True).start()
    return get_job(job_id)


@app.get("/api/jobs")
def list_jobs():
    with lock:
        rows = con.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 20").fetchall()
    return [job_progress(dict(r)) for r in rows]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int):
    with lock:
        r = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if r is None:
        raise HTTPException(404, "job not found")
    return job_progress(dict(r))


# ---------------------------------------------------------------- static files
# Windows can map .js to text/plain in the registry, which browsers refuse for module scripts
for ext, mime in ((".js", "text/javascript"), (".css", "text/css"), (".mp4", "video/mp4"), (".gif", "image/gif")):
    mimetypes.add_type(mime, ext)
if REPORT.exists():
    app.mount("/media", StaticFiles(directory=REPORT), name="media")
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="site")
