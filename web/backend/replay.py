"""Compact top-down replays from the episode recordings of scripts/view.py.

logs/replay_l{level}_s{seed}{variant}.npz (qpos, mocap_pos, mocap_quat, text, res, command at 50 Hz) ->
web/backend/cache/replay_{key}.json: 10 Hz robot x/y/yaw, can x/y/z, person and pet x/y(/yaw), task state,
plus the floorplan of that episode (walls / furniture of randomize_layout(rng(1000 + seed)), places, sink).

python -m web.backend.replay            # convert every recording (cached by npz mtime)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from .db import ROOT, connect

sys.path.insert(0, str(ROOT))

LOGS = ROOT / "logs"
CACHE = Path(__file__).resolve().parent / "cache"
ADDR_FILE = CACHE / "model_addr.json"
CTRL_HZ = 50
OUT_HZ = 10
NAME_RE = re.compile(r"replay_l(\d+)_s(\d+)(.*)\.npz$")


def model_addr() -> dict:
    """qpos addresses of the robot's free base and the can, mocap ids of person / pet (one model build, cached)."""
    if ADDR_FILE.exists():
        return json.loads(ADDR_FILE.read_text())
    import mujoco
    from homebot.env import HomeEnv
    env = HomeEnv(level=3, seed=1000)
    m = env.m
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    addr = {"base": int(m.jnt_qposadr[jid]), "cup": int(env.cup_qadr), "person": int(env.person_mocap),
            "pet": int(env.pet_mocap), "nq": int(m.nq)}
    env.close()
    CACHE.mkdir(exist_ok=True)
    ADDR_FILE.write_text(json.dumps(addr))
    return addr


def parse_name(path: Path):
    """(level, seed, variant) of a recording, e.g. replay_l2_s2_elderly_sink+0.116.npz -> (2, 2, '_elderly_sink+0.116')."""
    m = NAME_RE.search(path.name)
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else None


def replay_key(level: int, seed: int, variant: str = "") -> str:
    return f"l{level}_s{seed}{variant}"


def variant_info(variant: str) -> dict:
    m = re.search(r"_sink([+-][\d.]+)", variant)
    return {"sink_offset": float(m.group(1)) if m else 0.0,
            "person_profile": "elderly" if "_elderly" in variant else "adult",
            "policy": (re.search(r"_pol-([^_]+)", variant) or [None, "expert"])[1]}   # see run_episode.py


def yaw_of(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def floorplan(seed: int, sink_offset: float = 0.0) -> dict:
    from homebot import apartment as apt
    items = apt.randomize_layout(np.random.default_rng(1000 + seed))
    shapes = []
    for it in items:
        s = {"name": it.name, "kind": it.kind, "x": round(it.pos[0], 3), "y": round(it.pos[1], 3),
             "h": it.h, "movable": it.movable, "wall": it.name.startswith("wall")}
        if it.kind == "box":
            s.update(hx=it.size[0], hy=it.size[1], yaw=round(float(it.yaw), 4))
        else:
            s.update(r=it.size[0])
        shapes.append(s)
    places = {k: {"x": v["pos"][0], "y": v["pos"][1], "yaw": float(v["yaw"])} for k, v in apt.PLACES.items()}
    sx, sy = apt.SINK_CENTER + np.array([sink_offset, 0.0])
    return {"width": apt.WIDTH, "height": apt.HEIGHT, "items": shapes, "places": places,
            "sink": {"x": float(sx), "y": float(sy), "hx": float(apt.SINK_HALF[0]), "hy": float(apt.SINK_HALF[1])},
            "doorways": [[float(d[0]), float(d[1])] for d in apt.DOORWAYS],
            "robot_radius": apt.ROBOT_RADIUS, "person_radius": apt.PERSON_RADIUS,
            "pet_half": list(apt.PET_HALF)}


def convert(path: Path) -> dict:
    level, seed, variant = parse_name(path)
    a = model_addr()
    z = np.load(path, allow_pickle=False)
    step = CTRL_HZ // OUT_HZ
    idx = np.arange(0, len(z["qpos"]), step)
    if idx[-1] != len(z["qpos"]) - 1:
        idx = np.append(idx, len(z["qpos"]) - 1)
    q, mp, mq = z["qpos"][idx], z["mocap_pos"][idx], z["mocap_quat"][idx]
    b, c = a["base"], a["cup"]
    texts = [str(t) for t in z["text"][idx]]
    states, safety = [], []
    for t in texts:
        st, _, sf = t.partition("safety:")
        states.append(st.strip())
        safety.append(sf.strip())
    labels = sorted(set(states))
    slabels = sorted(set(safety))
    r2 = lambda v: np.round(v, 3).tolist()
    person = mp[:, a["person"], :2]
    pet = mp[:, a["pet"], :2]
    info = variant_info(variant)
    out = {
        "key": replay_key(level, seed, variant), "level": level, "seed": seed, "variant": variant, **info,
        "hz": OUT_HZ, "n": len(idx), "duration_s": round((len(z["qpos"]) - 1) / CTRL_HZ, 2),
        "result": str(z["res"]), "command": str(z["command"]),
        "robot": {"x": r2(q[:, b]), "y": r2(q[:, b + 1]), "yaw": r2(yaw_of(q[:, b + 3:b + 7]))},
        "can": {"x": r2(q[:, c]), "y": r2(q[:, c + 1]), "z": r2(q[:, c + 2])},
        # mocap bodies parked far outside the apartment when absent (level 1: no person, < 3: no pet)
        "person": None if (person < -5).all() else {"x": r2(person[:, 0]), "y": r2(person[:, 1])},
        "pet": None if (pet < -5).all() else {"x": r2(pet[:, 0]), "y": r2(pet[:, 1]),
                                              "yaw": r2(yaw_of(mq[:, a["pet"]]))},
        "state_labels": labels, "state": [labels.index(s) for s in states],
        "safety_labels": slabels, "safety": [slabels.index(s) for s in safety],
        "floorplan": floorplan(seed, info["sink_offset"]),
    }
    return out


def json_path(key: str) -> Path:
    return CACHE / f"replay_{key}.json"


def sync(con=None, keys: list[str] | None = None) -> list[dict]:
    """Convert new / changed recordings, register them in the replays table; return the replay list."""
    con = con or connect()
    CACHE.mkdir(exist_ok=True)
    seen = []
    for p in sorted(LOGS.glob("replay_l*_s*.npz")):
        parsed = parse_name(p)
        if parsed is None:
            continue
        level, seed, variant = parsed
        key = replay_key(level, seed, variant)
        if keys is not None and key not in keys:
            continue
        seen.append(key)
        mtime = p.stat().st_mtime
        row = con.execute("SELECT npz_mtime FROM replays WHERE key = ?", (key,)).fetchone()
        jp = json_path(key)
        if row and row["npz_mtime"] == mtime and jp.exists():
            continue
        try:
            data = convert(p)
        except Exception as e:  # half-written or foreign file
            print(f"skip {p.name}: {e}", file=sys.stderr)
            continue
        jp.write_text(json.dumps(data, separators=(",", ":")))
        with con:
            con.execute("INSERT OR REPLACE INTO replays(key, level, seed, variant, npz, npz_mtime, json_path, result,"
                        " duration_s) VALUES (?,?,?,?,?,?,?,?,?)",
                        (key, level, seed, variant, p.name, mtime, str(jp), data["result"], data["duration_s"]))
    if keys is None:
        with con:
            for r in con.execute("SELECT key FROM replays").fetchall():
                if r["key"] not in seen:
                    con.execute("DELETE FROM replays WHERE key = ?", (r["key"],))
    return list_replays(con)


def list_replays(con) -> list[dict]:
    rows = con.execute("SELECT key, level, seed, variant, npz, result, duration_s FROM replays "
                       "ORDER BY level, seed, variant").fetchall()
    return [{**dict(r), **variant_info(r["variant"])} for r in rows]


def load(key: str) -> dict | None:
    jp = json_path(key)
    return json.loads(jp.read_text()) if jp.exists() else None


if __name__ == "__main__":
    for r in sync():
        print(r)
