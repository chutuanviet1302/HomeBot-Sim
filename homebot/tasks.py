"""Task orchestrator: fixed command table -> state machine.

Only one branch drives the robot at a time: while navigating the planner sends base
velocities; once aligned in front of the table/sink the manipulation policy sends
upper-body commands.
"""
from __future__ import annotations

import numpy as np

from . import apartment as apt
from .env import CARRY_HIGH
from .expert import Expert
from .nav.navigator import Navigator

COMMANDS = {
    "go to the kitchen": [("NAVIGATE_TO", "kitchen")],
    "go to the living room": [("NAVIGATE_TO", "living_room")],
    # ALIGN is part of each manipulation skill: the robot steps until the perceived object (or the drop spot)
    # sits where the hand works best, then stands still
    "bring the can to the sink": [("NAVIGATE_TO", "table"), ("MANIPULATE", "pick_cup"),
                                  ("NAVIGATE_TO", "sink"), ("MANIPULATE", "place_cup")],
}


class TaskRunner:
    def __init__(self, env, policy=None, safety=True, timeout_nav=180.0):
        self.env = env
        self.nav = Navigator(env, safety=safety)
        self.policy = policy or Expert(env, rng=env.rng)   # anything with pick_cup()/place_cup()
        self.timeout_nav = timeout_nav
        self.trace = []
        self.state = "IDLE"
        self._aligned_at = {}

    def run(self, command):
        if command not in COMMANDS:
            raise ValueError(f"unknown command {command!r}; known: {list(COMMANDS)}")
        for state, arg in COMMANDS[command] + [("DONE", None)]:
            t0 = self.env.t
            self.state = f"{state}({arg})" if arg else state
            ok = True
            if state == "NAVIGATE_TO":
                place = apt.PLACES[arg]
                carrying = self.env.grasped
                if carrying:
                    self.env.wrist_pos = CARRY_HIGH.copy()
                goal = place.get("pre", place["pos"])
                ok = self.nav.go(goal, self.timeout_nav, carrying=carrying, tol=0.15)["success"]
            elif state == "MANIPULATE":
                ok = getattr(self.policy, arg)()
                if ok:
                    # step back 0.35 m from the table / counter edge (straight, heading kept, LiDAR clearance and the
                    # safety layer active); the next NAVIGATE_TO plans from wherever the robot then is
                    x, y, th = self.env.pose
                    back = np.array([x, y]) - 0.35 * np.array([np.cos(th), np.sin(th)])
                    self.nav.straight_to(back, 10.0, carrying=self.env.grasped, tol=0.08)
            self.trace.append({"state": state, "arg": arg, "ok": bool(ok), "t": round(self.env.t - t0, 2)})
            if not ok:
                return {"success": False, "failed_at": f"{state}({arg})", "trace": self.trace,
                        "time": self.env.t, **self._safety()}
        return {"success": True, "failed_at": None, "trace": self.trace, "time": self.env.t, **self._safety()}

    def _safety(self):
        return {k: v for k, v in self.env.stats.items() if not k.startswith("_")}
