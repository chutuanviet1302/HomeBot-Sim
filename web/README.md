# HomeBot-Sim results website

Full-stack site for the HomeBot-Sim results: FastAPI + SQLite backend (`web/backend/`), React + Vite + TypeScript
frontend (`web/frontend/`, hand-written SVG charts, Vietnamese UI). All commands run from the repository root.

Pages: **Tổng quan** (pain point, KPI cards, E4 safety comparison, E1 levels, expert vs ACT, E5),
**Thí nghiệm** (filterable / sortable runs table, run details, CSV export), **Xem lại** (top-down replay of a
recorded episode: robot / person / pet / can trails, scrubber, speed, task state, demo video),
**Chạy mô phỏng** (start a headless episode, progress bar, open its replay; hidden in the static build).

## Run

```bash
python -m web.backend.ingest                                  # results/*.csv -> web/homebot.db (idempotent)
npm --prefix web/frontend install
npm --prefix web/frontend run build                           # -> web/frontend/dist, served by the backend at /
python -m uvicorn web.backend.app:app --port 8000             # http://localhost:8000
```

Development (hot reload; Vite proxies `/api` and `/media` to port 8000):

```bash
python -m uvicorn web.backend.app:app --port 8000
npm --prefix web/frontend run dev                             # http://localhost:5173
```

`.claude/launch.json` has both as `homebot-api` and `homebot-web`.

Re-run `python -m web.backend.ingest` after new result CSVs appear (or delete `web/homebot.db`; the server ingests
on start when the database is empty). Recordings in `logs/replay_*.npz` are picked up automatically.

## Static build (no server)

```bash
python -m web.backend.export_static                           # GET responses -> web/frontend/public/data/*.json, media -> public/media/
npm --prefix web/frontend run build:static                    # -> web/frontend/dist-static (relative paths, open from any folder)
```

`export_static` leaves out the 31 MB `report/demo_full_task.mp4` so the build stays ~8 MB; add `--with-video` to
include it. The static build reads `data/*.json`, hides the Run page and works from any sub-path (hash routing).

## Data sources

| Experiment | File(s) | Notes |
|---|---|---|
| E1-L1/L2/L3, E4 | `results/e1_e4_nav.csv` | E1 = safety on, levels 1-3; E4 = level 2 with the safety layer off |
| E2-pick, E2-place | `results/e2_manip_<policy>.csv` | split by the `skill` column |
| E2-place-sink-moved | `results/e2_place_sink_moved.csv` | |
| E3 | `results/e3_full_level2_<policy>.csv` | `expert`, `act`, ... (the `policy` column wins) |
| E3-sink-moved | `results/e3_full_level2_sink_moved.csv` | |
| E5 | `results/e5_elderly.csv` (optional) | grouped by policy and any `person` / `safety` / `condition` column |

Common columns go to `runs` (see `web/backend/db.py`); every other CSV column is kept in `runs.extra` (JSON).
Missing files are skipped and the related sections show a placeholder.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/kpi` | KPI cards, E4 safety comparison, E1 levels, per-policy success, E5 |
| GET | `/api/experiments` | experiments with summaries |
| GET | `/api/runs?exp=&policy=&success=` | runs (filters optional) |
| GET | `/api/runs/{id}` | one run |
| GET | `/api/replays` | recorded episodes in `logs/` (converted and cached on first request) |
| GET | `/api/replays/{level}/{seed}?variant=` | 10 Hz trajectory + floorplan (`variant` e.g. `_elderly`, `_sink+0.116`) |
| GET | `/api/media` | available media under `/media` (`report/demo.gif`, `demo_full_task.mp4`, `figures/*`) |
| GET | `/api/policies` | policies the Run page offers (`expert`, `act:<checkpoint>.pt` found in the repo root, `logs/`, `checkpoints/`) |
| POST | `/api/jobs` | `{level, seed, sink_range, person, policy}`: queue a headless episode (one at a time) |
| GET | `/api/jobs`, `/api/jobs/{id}` | job status, progress (parsed from the log), log tail, replay key |

A job runs `web/backend/run_episode.py` (uses `record` / `save_cache` of `scripts/view.py`, no viewer window) and
writes `logs/replay_l{level}_s{seed}[_elderly][_sink±x][_pol-<policy>].npz`; a level 1 episode takes ~2 min,
level 2 ~3-4 min. Logs go to `web/backend/jobs/`, converted replays to `web/backend/cache/`.
