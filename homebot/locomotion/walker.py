"""Ported from the HomeHand project (robot_g1_inspirehand/homehand/locomotion/walker.py).

Walking on the G1's own legs with Unitree's pretrained locomotion policy (unitree_rl_gym, BSD-3).

The policy (`deploy/pre_train/g1/motion.pt`, trained in Isaac Gym for the 12-DoF G1) takes a velocity command
(forward, sideways, yaw rate) and proprioception, and outputs 12 leg joint targets at 50 Hz; a PD loop turns them
into torques at every 500 Hz physics step - exactly as in Unitree's own MuJoCo deployment
(`deploy/deploy_mujoco/deploy_mujoco.py`, `configs/g1.yaml`). Here the upper body carries two Inspire hands and
the arms move while the legs work: the policy keeps the balance (tested: walking, turning, standing while the arms
reach). Standing still, it keeps stepping in place and drifts ~2.5 cm/s, so `hold` closes a loop on the base pose
(velocity commands towards the stop point). That still sways the pelvis by +-7 cm, too much for the hands.

`stand` therefore switches to a balance stand without stepping (as the real G1's "balance stand" mode): once both
feet are on the floor, the legs blend to the policy's standing posture and hold it with stiff PD, gravity
compensation, and ankle / hip torques against the torso's tilt. Standing 30 s with the arms reaching: sideways
sway < 1 cm, tilt < 3 deg, a slow lean of up to 5 cm with the arms (which the arm IK absorbs). Freezing the legs
mid-step instead (one foot in the air) made the robot fall within a second.
"""

from __future__ import annotations

import mujoco
import numpy as np
import yaml

from pathlib import Path

RL_DIR = Path(__file__).resolve().parents[2] / "third_party" / "unitree_rl_gym"
LEG_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right")
              for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
MAX_CMD = np.array([0.5, 0.3, 0.8])          # forward m/s, sideways m/s, yaw rad/s
HOLD_KP = np.array([1.5, 1.5, 2.0])
HOLD_KI = np.array([2.0] * 2 + [0.5])
HOLD_I_MAX = np.array([0.15, 0.15, 0.3])
HOLD_MAX = np.array([0.3, 0.3, 0.5])
STAND_KP_SCALE, STAND_KD_SCALE = 3.0, 2.0     # x the policy's PD gains
TILT_KP, TILT_KD = 300.0, 30.0                # N m per rad, per rad/s
ANKLE_SHARE, HIP_SHARE = 0.5, 0.3
STAND_BLEND_S = 1.0
STAND_TARGET = "default"
STAND_FEET_DX = 0.02
STAND_FEET_W = 0.248                     # m: feet width of the default posture
STAND_FEET_DW = 0.015
STAND_FADE_S = 1.5
STAND_MAX_V = 0.08            # m/s: switch to standing only once the pelvis has (nearly) stopped


def yaw_of(quat) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class Walker:
    """Leg controller: `set_velocity(v)` or `hold(x, y, yaw)`, then call `physics_step()` before every mj_step."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        import torch
        torch.set_num_threads(1)          # deterministic, and the evaluations run one process per core
        self.torch = torch
        cfg = yaml.safe_load(open(RL_DIR / "deploy/deploy_mujoco/configs/g1.yaml"))
        self.policy = torch.jit.load(str(RL_DIR / "deploy/pre_train/g1/motion.pt"))
        self.m, self.d = model, data
        self.kp = np.array(cfg["kps"]); self.kd = np.array(cfg["kds"])
        self.default = np.array(cfg["default_angles"])
        self.decimation = int(cfg["control_decimation"])
        self.ang_vel_scale, self.dof_pos_scale = cfg["ang_vel_scale"], cfg["dof_pos_scale"]
        self.dof_vel_scale, self.action_scale = cfg["dof_vel_scale"], cfg["action_scale"]
        self.cmd_scale = np.array(cfg["cmd_scale"])
        self.qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in LEG_JOINTS])
        self.dofadr = np.array([model.jnt_dofadr[model.joint(j).id] for j in LEG_JOINTS])
        self.act = np.array([model.actuator(j.replace("_joint", "_motor")).id for j in LEG_JOINTS])
        self.base_q = model.jnt_qposadr[model.joint("floating_base_joint").id]
        self.base_v = model.jnt_dofadr[model.joint("floating_base_joint").id]
        self.foot_l, self.foot_r = model.body("left_ankle_roll_link").id, model.body("right_ankle_roll_link").id
        self.feet = {self.foot_l, self.foot_r}
        self.lo = model.actuator_ctrlrange[self.act, 0].copy()
        self.hi = model.actuator_ctrlrange[self.act, 1].copy()
        self.reset()

    def reset(self) -> None:
        self.mode = "walk"
        try:
            self.d.eq_active[self.m.equality("base_lock").id] = 0
        except KeyError:
            pass
        self.action = np.zeros(12)
        self.target = self.default.copy()
        self.cmd = np.zeros(3)
        self.goal = None              # (x, y, yaw) to hold, or None
        self.integral = np.zeros(3)
        self.mode = "walk"            # walk | to_stand (waiting for both feet down) | stand
        self.stand_from = None
        self.stand_k = 0
        self.count = 0
        self.obs = np.zeros(47, dtype=np.float32)

    # ------------------------------------------------------------------ commands
    def set_velocity(self, v) -> None:
        self.goal = None
        if self.mode not in ("walk",):
            self.walk()
        self.cmd = np.clip(np.asarray(v, float), -MAX_CMD, MAX_CMD)

    def hold(self, x: float, y: float, yaw: float) -> None:
        goal = (float(x), float(y), float(yaw))
        if goal != self.goal:
            self.integral = np.zeros(3)
        self.goal = goal

    def stand(self) -> None:
        """Stop stepping: balance stand as soon as both feet are on the floor."""
        if self.mode == "walk":
            self.mode = "to_stand"

    def lock(self) -> None:
        """Weld the pelvis where it stands and hold the legs (the base-lock simplification, see HomeEnv)."""
        m, d = self.m, self.d
        eid = m.equality("base_lock").id
        p = d.qpos[self.base_q:self.base_q + 3].copy()
        q = d.qpos[self.base_q + 3:self.base_q + 7].copy()
        qinv = np.zeros(4)
        mujoco.mju_negQuat(qinv, q)
        rel = np.zeros(3)
        mujoco.mju_rotVecQuat(rel, -p, qinv)          # the world frame seen from the pelvis
        m.eq_data[eid, 0:3] = 0.0
        m.eq_data[eid, 3:6] = rel
        m.eq_data[eid, 6:10] = qinv
        m.eq_data[eid, 10] = 1.0
        d.eq_active[eid] = 1
        self.lock_q = d.qpos[self.qadr].copy()
        self.mode = "locked"

    def unlock(self) -> None:
        self.d.eq_active[self.m.equality("base_lock").id] = 0
        self.walk()

    def walk(self) -> None:
        self.mode = "walk"
        self.reset_policy_state()

    def reset_policy_state(self) -> None:
        self.action = np.zeros(12)
        self.target = self.default.copy()

    def _feet_down(self) -> bool:
        m, d = self.m, self.d
        down = set()
        for i in range(d.ncon):
            c = d.contact[i]
            for g in (c.geom1, c.geom2):
                b = m.geom_bodyid[g]
                if b in self.feet:
                    down.add(b)
        return len(down) == 2

    def _feet_dx(self) -> float:
        lf, rf = self.d.xpos[self.foot_l], self.d.xpos[self.foot_r]
        _, yaw = self.base_pose()
        return float(np.cos(yaw) * (lf[0] - rf[0]) + np.sin(yaw) * (lf[1] - rf[1]))

    def _stance_for_feet(self) -> np.ndarray:
        """Leg joint targets that keep both feet where they stand and put the pelvis upright above the middle of
        them, at the default posture's height above the feet (blending to the default joint angles with the feet
        planted 17-21 cm apart instead of its 25 cm pushed the body sideways and it fell)."""
        m, d = self.m, self.d
        sd = mujoco.MjData(m)
        sd.qpos[:] = d.qpos
        mujoco.mj_kinematics(m, sd)
        feet = {f: (d.xpos[f].copy(), d.xquat[f].copy()) for f in (self.foot_l, self.foot_r)}
        mid = 0.5 * (feet[self.foot_l][0] + feet[self.foot_r][0])
        _, yaw = self.base_pose()
        # pelvis height above the feet in the default posture
        dq = mujoco.MjData(m)
        dq.qpos[:] = d.qpos
        dq.qpos[self.base_q:self.base_q + 7] = [0, 0, 1, 1, 0, 0, 0]
        dq.qpos[self.qadr] = self.default
        mujoco.mj_kinematics(m, dq)
        h = 1.0 - 0.5 * (dq.xpos[self.foot_l][2] + dq.xpos[self.foot_r][2])
        fwd = 0.5 * (dq.xpos[self.foot_l][0] + dq.xpos[self.foot_r][0])     # feet ahead of the pelvis by default
        c, s = np.cos(yaw), np.sin(yaw)
        sd.qpos[self.base_q:self.base_q + 3] = [mid[0] - c * fwd, mid[1] - s * fwd, mid[2] + h]
        sd.qpos[self.base_q + 3:self.base_q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        sd.qpos[self.qadr] = self.default
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(60):
            mujoco.mj_kinematics(m, sd)
            mujoco.mj_comPos(m, sd)
            for f in (self.foot_l, self.foot_r):
                tp, tq = feet[f]
                ep = tp - sd.xpos[f]
                er = np.zeros(3)
                qinv = np.zeros(4)
                mujoco.mju_negQuat(qinv, sd.xquat[f])
                qd = np.zeros(4)
                mujoco.mju_mulQuat(qd, tq, qinv)
                mujoco.mju_quat2Vel(er, qd, 1.0)
                mujoco.mj_jacBody(m, sd, jp, jr, f)
                J = np.vstack([jp, jr])[:, self.dofadr]
                cols = [i for i in range(12) if np.any(J[:, i])]
                Jc = J[:, cols]
                dqv = Jc.T @ np.linalg.solve(Jc @ Jc.T + 1e-4 * np.eye(6), np.r_[ep, er])
                sd.qpos[self.qadr[cols]] += dqv
        lo = m.jnt_range[[m.joint(j).id for j in LEG_JOINTS], 0]
        hi = m.jnt_range[[m.joint(j).id for j in LEG_JOINTS], 1]
        return np.clip(sd.qpos[self.qadr].copy(), lo, hi)

    def _feet_level(self) -> bool:
        """Feet side by side (neither ahead) at about the standing width: the default posture fits them."""
        lf, rf = self.d.xpos[self.foot_l], self.d.xpos[self.foot_r]
        _, yaw = self.base_pose()
        c, s = np.cos(yaw), np.sin(yaw)
        dx = c * (lf[0] - rf[0]) + s * (lf[1] - rf[1])
        dy = -s * (lf[0] - rf[0]) + c * (lf[1] - rf[1])
        return abs(dx) < STAND_FEET_DX and abs(dy - STAND_FEET_W) < STAND_FEET_DW

    def _calm(self) -> bool:
        v = self.d.qvel[self.base_v:self.base_v + 6]
        return float(np.linalg.norm(v[:2])) < STAND_MAX_V and float(np.linalg.norm(v[3:5])) < 0.3

    def base_pose(self) -> tuple[np.ndarray, float]:
        q = self.d.qpos
        return q[self.base_q:self.base_q + 3].copy(), yaw_of(q[self.base_q + 3:self.base_q + 7])

    def place(self, x: float, y: float, yaw: float) -> None:
        """Put the robot standing at (x, y) facing `yaw` (at reset)."""
        q = self.d.qpos
        q[self.base_q:self.base_q + 3] = [x, y, 0.793]
        q[self.base_q + 3:self.base_q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        q[self.qadr] = self.default
        self.d.qvel[self.base_v:self.base_v + 6] = 0.0
        self.reset()

    # ------------------------------------------------------------------ control
    def _hold_cmd(self) -> np.ndarray:
        pos, yaw = self.base_pose()
        gx, gy, gyaw = self.goal
        ex, ey = gx - pos[0], gy - pos[1]
        c, s = np.cos(yaw), np.sin(yaw)
        fx, fy = c * ex + s * ey, -s * ex + c * ey              # error in the base frame
        e = np.array([fx, fy, wrap(gyaw - yaw)])
        # PI: the policy barely reacts to commands of a few cm/s, so a pure P loop settled 10-17 cm off
        dt = self.decimation * self.m.opt.timestep
        self.integral = np.clip(self.integral + e * dt, -HOLD_I_MAX, HOLD_I_MAX)
        return np.clip(HOLD_KP * e + HOLD_KI * self.integral, -HOLD_MAX, HOLD_MAX)

    def physics_step(self) -> None:
        if self.mode == "holding":           # HomeBot: stepping in place on a PI pose hold
            self._policy_step()
            return
        if self.mode == "to_stand" and self._feet_down() and self._calm() and abs(self._feet_dx()) < STAND_FEET_DX:
            self.mode, self.stand_from, self.stand_k = "stand", self.d.qpos[self.qadr].copy(), 0
            self.stand_target = self._stance_for_feet()
        if self.mode == "locked":
            d = self.d
            tau = (self.lock_q - d.qpos[self.qadr]) * self.kp - d.qvel[self.dofadr] * self.kd
            d.ctrl[self.act] = np.clip(tau, self.lo, self.hi)
            return
        if self.mode == "stand":
            u = min(1.0, self.stand_k * self.m.opt.timestep / STAND_FADE_S)
            if u >= 1.0:
                self._stand_step()
                return
            # fade: the walking policy (stepping in place, zero command) keeps the balance while the stand
            # controller takes over; an abrupt switch carried the gait's momentum into the stance
            self._stand_step()
            self.count -= 1                       # (one physics step: the gait clock advances once)
            tau_stand = self.d.ctrl[self.act].copy()
            self.cmd = np.zeros(3)
            self._policy_step(hold_goal=False)
            tau_walk = self.d.ctrl[self.act].copy()
            self.d.ctrl[self.act] = np.clip((1 - u) * tau_walk + u * tau_stand, self.lo, self.hi)
            return
        self._policy_step()

    def _policy_step(self, hold_goal: bool = True) -> None:
        d = self.d
        q, dq = d.qpos[self.qadr], d.qvel[self.dofadr]
        d.ctrl[self.act] = (self.target - q) * self.kp - dq * self.kd
        if self.count % self.decimation == 0:
            cmd = self._hold_cmd() if (self.goal is not None and hold_goal) else self.cmd
            qw, qx, qy, qz = d.qpos[self.base_q + 3:self.base_q + 7]
            grav = np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])
            phase = (self.count * self.m.opt.timestep) % 0.8 / 0.8
            o = self.obs
            o[:3] = d.qvel[self.base_v + 3:self.base_v + 6] * self.ang_vel_scale
            o[3:6] = grav
            o[6:9] = cmd * self.cmd_scale
            o[9:21] = (q - self.default) * self.dof_pos_scale
            o[21:33] = dq * self.dof_vel_scale
            o[33:45] = self.action
            o[45:47] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
            with self.torch.no_grad():
                self.action = self.policy(self.torch.from_numpy(o).unsqueeze(0)).numpy().squeeze().astype(float)
            self.target = self.action * self.action_scale + self.default
        self.count += 1

    def _stand_step(self) -> None:
        d = self.d
        q, dq = d.qpos[self.qadr], d.qvel[self.dofadr]
        u = min(1.0, self.stand_k * self.m.opt.timestep / STAND_BLEND_S)
        target = self.stand_from + (self.stand_target - self.stand_from) * (u * u * (3 - 2 * u))
        tau = (target - q) * self.kp * STAND_KP_SCALE - dq * self.kd * STAND_KD_SCALE + d.qfrc_bias[self.dofadr]
        qw, qx, qy, qz = d.qpos[self.base_q + 3:self.base_q + 7]
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        wx, wy = d.qvel[self.base_v + 3], d.qvel[self.base_v + 4]
        tp, tr = TILT_KP * pitch + TILT_KD * wy, TILT_KP * roll + TILT_KD * wx
        for side in (0, 6):   # [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll] per leg
            tau[side + 4] += ANKLE_SHARE * tp
            tau[side + 5] -= ANKLE_SHARE * tr
            tau[side + 0] += HIP_SHARE * tp
            tau[side + 1] -= HIP_SHARE * tr
        d.ctrl[self.act] = np.clip(tau, self.lo, self.hi)
        self.stand_k += 1
        self.count += 1

    def fallen(self) -> bool:
        return float(self.d.qpos[self.base_q + 2]) < 0.5
