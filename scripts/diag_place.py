"""Debug helper: one expert place from the holding state, with a phase log."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_manip import perturbed, put_cup_in_hand  # noqa: E402
from homebot import apartment as apt  # noqa: E402
from homebot import expert as X  # noqa: E402
from homebot.env import HomeEnv  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(50_000 + seed)
env = HomeEnv(level=1, seed=seed)
start = perturbed("sink", rng)
start[:2] -= 0.12 * np.array([np.cos(start[2]), np.sin(start[2])])
env.reset(start, arm_ready=True)
from homebot.env import quat_to_mat as _q2m
print("held after init:", put_cup_in_hand(env, rng, start), "cup", env.cup_pose()[0].round(3), "up", _q2m(env.cup_pose()[1])[:, 2].round(2))
ex = X.Expert(env, rng=rng)
ex.debug = True
om = ex.move_hand


def mh(t, R, **kw):
    om(t, R, **kw)
    p, _ = env.hand_world()
    print(f"move palm err {np.linalg.norm(p - t):.3f} target {np.round(t, 3)} armc {env.stats['arm_contacts']} stops {env.stats['arm_stops']} up {_q2m(env.cup_pose()[1])[:, 2].round(2)} cup {env.cup_pose()[0].round(3)} grasped {env.grasped} contacts {env.cup_contacts()}")


ex.move_hand = mh
ok_al = ex.align  # log align result
ex.align = lambda *a, **k: print("align", ok_al(*a, **k), "pose", env.pose.round(3))
ok = ex.place_cup()
from homebot.env import quat_to_mat
cp, cq = env.cup_pose()
print("arm", {k: v for k, v in env.stats.items() if "arm" in k}, "ik_err", round(env.ik_err, 3))
print("place", ok, "cup", cp.round(3), "up", quat_to_mat(cq)[:, 2].round(3), "z_rest", round(apt.COUNTER_TOP + env.cup_h / 2, 3), "contacts", env.cup_contacts(), "sink", apt.SINK_CENTER)
