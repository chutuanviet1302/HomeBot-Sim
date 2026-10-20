"""Bridge between high-level commands and the whole-body controller.

Two commands reach the controller, exactly like the GEAR-SONIC keyboard/planner
interface:
  * base velocity (vx, vy, wz) in the robot frame       -> lower body (locomotion)
  * right-wrist pose in the pelvis frame + body height/lean + 6 finger closures
                                                         -> upper body / hand

`KinematicSonic` is the default backend: a kinematic proxy of what SONIC does
(tracks the velocity command with acceleration limits and a first-order lag, holds
balance by construction). It lets the whole navigation / task stack run on a laptop
without the GR00T-WholeBodyControl checkout.

`SonicOnnx` is the drop-in slot for the real policy (encoder/decoder/planner ONNX
from the GEAR-SONIC release). Its I/O must be matched against the official
`run_sim_loop.py` / VLA-inference tutorial before use, so it raises until wired up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VelLimits:
    vx_min: float = -0.3
    vx_max: float = 0.6
    vy_max: float = 0.2
    wz_max: float = 0.8
    acc: float = 0.5        # m/s^2 (smooth commands)
    brake: float = 2.5      # m/s^2 allowed when slowing down (emergency stop: the gait itself needs ~0.3 s)
    alpha: float = 1.2      # rad/s^2
    vx_max_carry: float = 0.25      # gentle gait while holding an object (less in-hand slip)


def clamp_cmd(cmd, lim: VelLimits, carrying=False):
    vx, vy, wz = cmd
    vmax = lim.vx_max_carry if carrying else lim.vx_max
    return np.array([np.clip(vx, lim.vx_min, vmax), np.clip(vy, -lim.vy_max, lim.vy_max),
                     np.clip(wz, -lim.wz_max, lim.wz_max)])


class KinematicSonic:
    """Velocity-tracking proxy for GEAR-SONIC (50 Hz)."""

    def __init__(self, dt=0.02, limits: VelLimits | None = None, tau=0.15):
        self.dt = dt
        self.lim = limits or VelLimits()
        self.tau = tau
        self.v = np.zeros(3)          # actual body-frame velocity
        self.max_acc_seen = 0.0

    def reset(self):
        self.v[:] = 0.0
        self.max_acc_seen = 0.0

    def step(self, pose, cmd, carrying=False):
        """pose = (x, y, yaw) in world; cmd = (vx, vy, wz) robot frame. Returns new pose."""
        target = clamp_cmd(cmd, self.lim, carrying)
        dv = (target - self.v) * min(1.0, self.dt / self.tau)
        braking = np.linalg.norm(target[:2]) < np.linalg.norm(self.v[:2])
        a_lin = (self.lim.brake if braking else self.lim.acc) * self.dt
        a_ang = self.lim.alpha * self.dt
        n = np.linalg.norm(dv[:2])
        if n > a_lin:
            dv[:2] *= a_lin / n
        dv[2] = np.clip(dv[2], -a_ang, a_ang)
        self.max_acc_seen = max(self.max_acc_seen, float(np.linalg.norm(dv[:2]) / self.dt))
        self.v = self.v + dv
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        x += (c * self.v[0] - s * self.v[1]) * self.dt
        y += (s * self.v[0] + c * self.v[1]) * self.dt
        th = (th + self.v[2] * self.dt + np.pi) % (2 * np.pi) - np.pi
        return np.array([x, y, th])

    def fell(self):
        # The proxy cannot fall; a real SONIC run checks pelvis height < 0.5 m.
        return False


class SonicOnnx:  # pragma: no cover - requires the GR00T-WholeBodyControl release
    """Slot for the real GEAR-SONIC ONNX policy (29-DoF G1, 50 Hz)."""

    def __init__(self, model_dir: str):
        import onnxruntime as ort  # noqa: F401
        self.model_dir = model_dir
        raise NotImplementedError(
            "Wire the encoder/decoder/planner ONNX I/O from GR00T-WholeBodyControl "
            "(gear_sonic deploy, run_sim_loop.py) here; see README 'Real SONIC'.")
