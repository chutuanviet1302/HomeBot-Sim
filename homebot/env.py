"""MuJoCo apartment environment: G1 proxy driven through the SONIC bridge, person and pet."""
from __future__ import annotations

import mujoco
import numpy as np

from . import apartment as apt
from . import g1 as g1mod
from .nav.astar import astar, inflate, path_to_waypoints
from .locomotion.walker import Walker
from .sonic_bridge import KinematicSonic, VelLimits

CTRL_DT = 0.02          # 50 Hz control (SONIC rate)
N_SUB = 10              # physics substeps (timestep 0.002, as the Menagerie G1)
VEL_KP = np.array([0.3, 0.3, 0.3])       # velocity tracking around the walking policy
VEL_KI = np.array([1.0, 1.5, 1.0])
VEL_I_MAX = np.array([0.15, 0.2, 0.3])
ARM_SWING_GAIN = 0.6       # shoulder pitch per rad of the opposite hip's pitch (human-like counter-swing)
ARM_SWING_MAX = 0.30       # rad
POSE_NOISE = 0.02       # m, Gaussian noise on the robot pose given to the planner
CARRY_DEFAULT = np.array([0.25, -0.20, 0.95])   # palm rest pose in the robot frame (empty hand)
CARRY_HIGH = np.array([0.25, -0.20, 1.03])      # palm pose while carrying: the held cup clears the 0.90 m counter
CARRY_APPROACH = np.array([0.30, -0.25, 1.08])  # last steps to the counter: the held object clears its edge


def yaw_quat(th):
    return np.array([np.cos(th / 2), 0, 0, np.sin(th / 2)])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def obb_circle_distance(center, yaw, half, p):
    """Distance from point p to an oriented 2D box (negative inside)."""
    d = np.asarray(p) - np.asarray(center)
    c, sn = np.cos(-yaw), np.sin(-yaw)
    lx, ly = c * d[0] - sn * d[1], sn * d[0] + c * d[1]
    q = np.array([abs(lx) - half[0], abs(ly) - half[1]])
    return float(np.linalg.norm(np.maximum(q, 0)) + min(max(q[0], q[1]), 0))


class Person:
    """Walks between random free goals at 0.8-1.2 m/s, about half the time cutting
    across the robot's route. It never yields to a *moving* robot (worst case for the
    planner); like a real person it does not walk straight into a robot that is
    standing still or backing away - it waits and then re-routes around it."""

    def __init__(self, rng, occ_true, robot_route=None):
        self.rng = rng
        self.occ = inflate(occ_true, apt.PERSON_RADIUS + 0.08)
        self.route = robot_route
        self.speed = rng.uniform(0.8, 1.2)
        self.pos = self._sample_free()
        self.wps, self.pause = [], 0.0
        self.vel = np.zeros(2)

    def _sample_free(self):
        H, W = self.occ.shape
        while True:
            p = np.array([self.rng.uniform(0.3, apt.WIDTH - 0.3), self.rng.uniform(0.3, apt.HEIGHT - 0.3)])
            if not apt.occupied(self.occ, p):
                return p

    def _new_goal(self, robot_xy):
        if self.route is not None and self.rng.random() < 0.5:
            # cut across the robot's route: aim through a random point of it to the far side
            a, b = np.asarray(self.route[0], float), np.asarray(self.route[1], float)
            via = a + self.rng.uniform(0.2, 0.9) * (b - a)
            g = via + (via - self.pos) * self.rng.uniform(0.5, 1.0)
            g = np.clip(g, 0.3, [apt.WIDTH - 0.3, apt.HEIGHT - 0.3])
            if not apt.occupied(self.occ, g):
                return g
        return self._sample_free()

    def step(self, dt, robot_xy, robot_passive=False):
        if robot_passive and robot_xy is not None and len(self.wps):
            d = self.wps[0] - self.pos
            nxt = self.pos + d / max(np.linalg.norm(d), 1e-6) * self.speed * dt
            gap = np.linalg.norm(nxt - robot_xy) - apt.PERSON_RADIUS - apt.ROBOT_RADIUS
            if gap < 0.10 and gap < np.linalg.norm(self.pos - robot_xy) - apt.PERSON_RADIUS - apt.ROBOT_RADIUS:
                self.vel[:] = 0
                self.blocked = getattr(self, "blocked", 0.0) + dt
                if self.blocked > 4.0:      # still no way past: step aside, away from the robot
                    away = self.pos - robot_xy
                    away = away / (np.linalg.norm(away) + 1e-6)
                    for d_ in (0.8, 0.5, 0.3):
                        g = self.pos + away * d_
                        if not apt.occupied(self.occ, g):
                            self.wps = [g]
                            self.pause = 2.0
                            break
                    self.blocked = 0.0
                    return
                if self.blocked > 1.5 and not getattr(self, "rerouted", False):   # re-route around the robot
                    self.rerouted = True
                    occ = self.occ.copy()
                    rr, cc = np.mgrid[0:occ.shape[0], 0:occ.shape[1]]
                    ctr = apt.cell_to_world(rr, cc)
                    occ |= np.linalg.norm(ctr - robot_xy, axis=-1) < apt.ROBOT_RADIUS + apt.PERSON_RADIUS + 0.1
                    path = astar(occ, self.pos, self.wps[-1])
                    for _ in range(10):  # no way around: turn back and go somewhere else
                        if path is not None:
                            break
                        g = self._sample_free()
                        if not occ[apt.world_to_cell(g)]:
                            path = astar(occ, self.pos, g)
                    self.wps = list(path_to_waypoints(path, 0.3)[1:]) if path else []
                    self.blocked = 0.0
                return
        self.blocked = 0.0
        self.rerouted = False
        if self.pause > 0:
            self.pause -= dt
            self.vel[:] = 0
            return
        if not len(self.wps):
            path = astar(self.occ, self.pos, self._new_goal(robot_xy))
            if path is None or len(path) < 2:
                self.pause = 0.5
                return
            self.wps = list(path_to_waypoints(path, 0.3)[1:])
            self.speed = self.rng.uniform(0.8, 1.2)
        tgt = self.wps[0]
        d = tgt - self.pos
        n = np.linalg.norm(d)
        step = self.speed * dt
        if n <= step:
            self.pos = tgt.copy()
            self.wps.pop(0)
            if not self.wps:
                self.pause = self.rng.uniform(0.0, 2.0)
        else:
            self.pos = self.pos + d / n * step
        self.vel = d / max(n, 1e-6) * self.speed


class Pet:
    """Random walk 0.5-1.5 m/s with sudden direction changes; stays in free space."""

    def __init__(self, rng, occ_true):
        self.rng = rng
        self.occ = inflate(occ_true, 0.2)
        while True:
            p = np.array([rng.uniform(0.5, 7.5), rng.uniform(0.5, 5.5)])
            if not apt.occupied(self.occ, p):
                break
        self.pos = p
        self.yaw = rng.uniform(-np.pi, np.pi)
        self.speed = rng.uniform(0.5, 1.5)
        self.timer = rng.uniform(0.5, 2.0)
        self.vel = np.zeros(2)

    def step(self, dt, robot_xy=None, robot_passive=False):
        self.timer -= dt
        if self.timer <= 0:
            self.yaw += self.rng.uniform(-2.5, 2.5)
            self.speed = self.rng.choice([0.0, self.rng.uniform(0.5, 1.5)], p=[0.15, 0.85])
            self.timer = self.rng.uniform(0.5, 2.0)
        for _ in range(8):
            v = self.speed * np.array([np.cos(self.yaw), np.sin(self.yaw)])
            nxt = self.pos + v * dt
            near_robot = (robot_passive and robot_xy is not None
                          and np.linalg.norm(nxt - robot_xy) < apt.ROBOT_RADIUS + 0.30
                          and np.linalg.norm(nxt - robot_xy) < np.linalg.norm(self.pos - robot_xy))
            if not apt.occupied(self.occ, nxt) and not near_robot:
                self.pos = nxt
                self.vel = v
                return
            self.yaw += self.rng.uniform(1.0, 3.0)
        self.vel = np.zeros(2)


class HomeEnv:
    """Unitree G1 (legged) + Inspire hands in the apartment.

    level 1: static furniture + displaced chairs; 2: + one person; 3: + person + pet.
    Locomotion: base velocity commands -> rate limiter -> Unitree's walking policy on the G1's legs.
    Manipulation: `stand_mode=True` puts the legs into the balance stand; the right palm target
    (`wrist_pos`/`wrist_quat` in the robot frame) is tracked by IK + arm actuators and the 6 Inspire
    joint targets in `fingers` (radians). Objects are held only through contact.
    """

    def __init__(self, level=1, seed=0, cup_type=None, with_toy=False, limits: VelLimits | None = None,
                 render=False, perception=True, relaxed_arms=True, sink_offset=0.0):
        self.level = level
        self.relaxed_arms = relaxed_arms      # walk with hanging, swinging arms; the hand gets ready when it works
        self.use_perception = perception      # object poses from the head RGB-D camera (homebot/perception.py)
        self._perception = None
        self.held_rel = None                  # object pose in the hand frame, measured when the hand closed
        self.rng = np.random.default_rng(seed)
        self.items = apt.randomize_layout(self.rng)
        self.cup_type = cup_type or self.rng.choice(list(apt.CUP_TYPES))
        col = self.rng.uniform(0.2, 0.95, 3)
        # ground truth of the sink (scene construction and success checks only; the robot perceives the basin)
        self.sink_center = apt.SINK_CENTER + np.array([sink_offset, 0.0])
        xml = apt.build_mjcf(self.items, self.cup_type, f"{col[0]:.2f} {col[1]:.2f} {col[2]:.2f} 1", with_toy,
                             proxy_robot=False, sink_center=self.sink_center)
        spec = mujoco.MjSpec.from_string(xml)
        g1mod.build_spec(spec)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.m.opt.noslip_iterations = 3      # stable frictional grasps (as in HomeHand)
        bid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
        gid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)
        jad = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.person_mocap = self.m.body_mocapid[bid("person")]
        self.pet_mocap = self.m.body_mocapid[bid("pet")]
        self.robot_body = bid("pelvis")
        self.cup_qadr = self.m.jnt_qposadr[jad("cup_free")]
        self.cup_dadr = self.m.jnt_dofadr[jad("cup_free")]
        self.cup_body = bid("cup")
        self.cup_h = apt.CUP_TYPES[self.cup_type]["h"]
        self.cup_r = apt.CUP_TYPES[self.cup_type]["r"]
        self.toy_qadr = self.m.jnt_qposadr[jad("toy_free")] if with_toy else None
        self.ctrl = g1mod.G1Controller(self.m)
        self.walker = Walker(self.m, self.d)
        under = lambda b: b == self.robot_body or (b > 0 and under(int(self.m.body_parentid[b])))
        self.robot_bodies = {b for b in range(self.m.nbody) if under(b)}
        hand_bodies = {b for b in self.robot_bodies if self.m.body(b).name.startswith("R_")}
        self.hand_geoms = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in hand_bodies}
        self.thumb_geoms = {g for g in self.hand_geoms if "thumb" in self.m.body(int(self.m.geom_bodyid[g])).name}
        self.feet = {bid("left_ankle_roll_link"), bid("right_ankle_roll_link")}
        self.floor = gid("floor")
        counter_parts = {gid("counter" + k) for k in ("_l", "_r", "_f", "_b")}
        self.furniture_geoms = {gid(it.name) for it in self.items} | counter_parts
        self.support_geoms = {gid("dining_table"), gid("counter"), self.floor} | counter_parts

        self.occ_true = apt.rasterize(self.items)
        self.sonic = KinematicSonic(CTRL_DT, limits or VelLimits(vx_max=0.5, vy_max=0.2, wz_max=0.8))
        self.person = None
        self.pet = None
        self.t = 0.0
        self.pose = np.zeros(3)
        self.pose_f = np.zeros(3)
        self.stand_mode = False
        self.wrist_pos = CARRY_DEFAULT.copy()
        self.wrist_quat = np.array([1.0, 0, 0, 0])
        self.fingers = g1mod.RELAXED_HAND.copy()
        self.vel_f = np.zeros(3)
        self.vel_i = np.zeros(3)
        self.hand_goal_world = None
        self.arm_q_goal = None
        self.arm_relaxed = relaxed_arms
        self._carry_key, self._carry_q = None, None
        self.ik_err = 0.0
        self.stats = {}
        jq = lambda n: self.m.jnt_qposadr[self.m.joint(n).id]
        self.hip_q = np.array([jq("left_hip_pitch_joint"), jq("right_hip_pitch_joint")])
        self.hip_mean = np.zeros(2)

    # ----------------------------------------------------------------- reset
    def reset(self, start_pose, robot_route=None, place_cup=True, cup_xy=None, arm_ready=False):
        """`arm_ready`: start with the hand already in the ready pose (manipulation trials that start at the
        table / counter, where raising a hanging arm would brush the edge); otherwise relaxed if enabled."""
        mujoco.mj_resetData(self.m, self.d)
        self.t = 0.0
        self.sonic.reset()
        self.vel_f = np.zeros(3)
        self.vel_i = np.zeros(3)
        self.hand_goal_world = None
        self.arm_q_goal = None
        self.stand_mode = False
        self.fingers = g1mod.RELAXED_HAND.copy()
        self.wrist_pos = CARRY_DEFAULT.copy()
        self.wrist_quat = np.array([1.0, 0, 0, 0])
        self.held_rel = None
        if self._perception is not None:
            self._perception.reset()
        self.walker.place(*start_pose)
        self.arm_relaxed = self.relaxed_arms and not arm_ready
        self._carry_key, self._carry_q = None, None
        self.ctrl.set_ready(self.d, relaxed=self.arm_relaxed)
        self.hip_mean = self.d.qpos[self.hip_q].copy()
        self.pose = np.array(start_pose, float)
        self.pose_f = self.pose.copy()
        self.person = self.pet = None
        if self.level >= 2:
            self.person = Person(self.rng, self.occ_true, robot_route)
            while np.linalg.norm(self.person.pos - self.pose[:2]) < 1.5:
                self.person.pos = self.person._sample_free()
        if self.level >= 3:
            self.pet = Pet(self.rng, self.occ_true)
            while np.linalg.norm(self.pet.pos - self.pose[:2]) < 1.5:
                self.pet = Pet(self.rng, self.occ_true)
        if cup_xy is None:
            cup_xy = apt.CUP_HOME + self.rng.uniform(-0.10, 0.10, 2) * np.array([1.0, 0.3])
        self.set_cup(np.array([cup_xy[0], cup_xy[1], apt.TABLE_TOP + self.cup_h / 2 + 0.002]),
                     yaw_quat(self.rng.uniform(-np.pi, np.pi)))
        self.stats = {"person_contacts": 0, "pet_contacts": 0, "furniture_contacts": 0,
                      "person_contacts_moving": 0, "pet_contacts_moving": 0,
                      "min_person_dist": np.inf, "min_pet_dist": np.inf, "_in_p": False, "_in_pet": False,
                      "_in_f": False, "fell": False, "arm_contacts": 0, "arm_stops": 0, "_in_arm": False,
                      "arm_paths_rejected": 0, "perc_err_mm": float("nan"), "perc_frames": 0,
                      "left_arm_contacts": 0, "_in_larm": False}
        self._apply_mocap()
        mujoco.mj_forward(self.m, self.d)
        if not self.arm_relaxed:
            # start with the arm already in the (collision-checked) carry configuration
            p_w, R_w = self.hand_target_world()
            q, _ = self.ctrl.solve_best(self.d, p_w, R_w)
            self.d.qpos[self.ctrl.ik_q] = q
            self.ctrl.q_tgt = q.copy()
        mujoco.mj_forward(self.m, self.d)
        for _ in range(10):           # 0.2 s on its feet before anything happens
            self.step(np.zeros(3))
        self.t = 0.0
        return self.observe()

    def set_cup(self, pos, quat):
        a = self.cup_qadr
        self.d.qpos[a:a + 3] = pos
        self.d.qpos[a + 3:a + 7] = quat
        self.d.qvel[self.cup_dadr:self.cup_dadr + 6] = 0

    # ----------------------------------------------------------------- helpers
    def _true_pose(self):
        p, yaw = self.walker.base_pose()
        return np.array([p[0], p[1], yaw])

    def est_pose(self):
        return self.pose + np.array([*self.rng.normal(0, POSE_NOISE, 2), self.rng.normal(0, 0.005)])

    def est_pose_avg(self):
        """Gait-averaged pose estimate: the pelvis sways +-3 cm with every step, fine alignment uses the mean."""
        return self.pose_f + np.array([*self.rng.normal(0, POSE_NOISE, 2), self.rng.normal(0, 0.005)])

    def vel_body(self):
        """Measured base velocity (vx, vy, wz) in the robot frame."""
        v = self.d.qvel[self.walker.base_v:self.walker.base_v + 6]
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[5]])

    def hand_target_world(self):
        """Palm target in world. While manipulating, the expert/policy fixes it in the world (`hand_goal_world`)
        so the arm compensates the slow sway of the standing robot; otherwise it is `wrist_pos`/`wrist_quat` in
        the robot frame ((x, y) and yaw of the pelvis, z from the floor)."""
        if self.hand_goal_world is not None:
            return self.hand_goal_world
        x, y, th = self.pose
        R = quat_to_mat(yaw_quat(th))
        return np.array([x, y, 0.0]) + R @ self.wrist_pos, R @ quat_to_mat(self.wrist_quat)

    def robot_to_world(self, p_r, quat_r=(1.0, 0, 0, 0)):
        x, y, th = self.pose
        R = quat_to_mat(yaw_quat(th))
        return np.array([x, y, 0.0]) + R @ np.asarray(p_r, float), R @ quat_to_mat(quat_r)

    def world_to_robot(self, p_w):
        x, y, th = self.pose
        R = quat_to_mat(yaw_quat(th))
        return R.T @ (np.asarray(p_w, float) - np.array([x, y, 0.0]))

    def hand_world(self):
        """Actual palm-site pose (position, quaternion) in world."""
        p, R = self.ctrl.site_pose(self.d)
        return p, mat_to_quat(R)

    def close(self):
        """Release the OpenGL renderer(s) (evaluations create many environments per process)."""
        if self._perception is not None:
            self._perception.r.close()
            self._perception = None

    @property
    def perception(self):
        """Head-camera object perception (created on first use; None if disabled)."""
        if self.use_perception and self._perception is None:
            from .perception import Perception
            self._perception = Perception(self)
        return self._perception if self.use_perception else None

    def cup_pose(self):
        a = self.cup_qadr
        return self.d.qpos[a:a + 3].copy(), self.d.qpos[a + 3:a + 7].copy()

    def cup_contacts(self):
        """(touching the thumb, touching another finger/palm, resting on a support surface)."""
        thumb = other = support = False
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if self.m.geom_bodyid[a] == self.cup_body:
                o = b
            elif self.m.geom_bodyid[b] == self.cup_body:
                o = a
            else:
                continue
            if o in self.thumb_geoms:
                thumb = True
            elif o in self.hand_geoms:
                other = True
            elif o in self.support_geoms:
                support = True
        return thumb, other, support

    @property
    def grasped(self):
        """Cup held by contact: touching the thumb and another finger, not resting on anything."""
        thumb, other, support = self.cup_contacts()
        return thumb and other and not support

    def _apply_mocap(self):
        if self.person is not None:
            self.d.mocap_pos[self.person_mocap] = [*self.person.pos, 0]
        else:
            self.d.mocap_pos[self.person_mocap] = [-10, -10, 0]
        if self.pet is not None:
            self.d.mocap_pos[self.pet_mocap] = [*self.pet.pos, 0]
            self.d.mocap_quat[self.pet_mocap] = yaw_quat(self.pet.yaw)
        else:
            self.d.mocap_pos[self.pet_mocap] = [-12, -12, 0]

    # ----------------------------------------------------------------- step
    def step(self, cmd, carrying=False):
        """Advance 20 ms. cmd = (vx, vy, wz) robot frame -> walking policy (ignored in stand mode)."""
        self.sonic.step(self.pose, cmd, carrying)          # rate / acceleration limiter of the command
        if self.stand_mode:
            # manipulation: step in place holding the pose for ~1 s, then lock the base (HomeHand's base-lock
            # simplification: stepping in place sways the pelvis ~16 cm, the balance stand never engages on
            # this model). Walking itself is always done by the policy.
            self.sonic.v[:] = 0
            if self.walker.mode == "walk":
                self.walker.hold(*self.pose)
                self.walker.mode = "holding"
                self.hold_t = self.t
            settled = np.linalg.norm(self.vel_f[:2]) < 0.03 and np.linalg.norm(self.vel_body()[:2]) < 0.08
            if self.walker.mode == "holding" and self.t - self.hold_t > 1.0 and (settled or self.t - self.hold_t > 3.0):
                self.walker.lock()
                self.stats["base_locks"] = self.stats.get("base_locks", 0) + 1
            self.vel_i[:] = 0
            self.vel_f += CTRL_DT / 0.4 * (self.vel_body() - self.vel_f)
        else:
            if self.walker.mode == "locked":
                self.walker.unlock()
            elif self.walker.mode == "holding":
                self.walker.mode = "walk"
            self.walker.set_velocity(self._track(self.sonic.v))
        if self.person is not None:
            self.person.step(CTRL_DT, self.pose[:2], not self._approaching(self.person.pos))
        if self.pet is not None:
            self.pet.step(CTRL_DT, self.pose[:2], not self._approaching(self.pet.pos))
        self._apply_mocap()
        swing_r, swing_l = self._arm_swing()
        self.ctrl.l_tgt = self.ctrl.l_tgt + np.clip(self.ctrl.l_rest + swing_l - self.ctrl.l_tgt,
                                                    -self.ctrl.max_dq, self.ctrl.max_dq)
        if self.arm_q_goal is not None:          # joint-space move planned beforehand (no IK at run time)
            self.ctrl.q_tgt = self.ctrl.q_tgt + np.clip(self.arm_q_goal - self.ctrl.q_tgt, -self.ctrl.max_dq,
                                                        self.ctrl.max_dq)
        elif self.arm_relaxed:                   # walking with an empty hand: arm hanging, swinging
            self.ctrl.q_tgt = self.ctrl.q_tgt + np.clip(self.ctrl.r_rest + swing_r - self.ctrl.q_tgt,
                                                        -self.ctrl.max_dq, self.ctrl.max_dq)
        elif carrying and self.hand_goal_world is None:
            # carrying: the arm is held still relative to the body (joints solved once for the carry pose).
            # Chasing the swaying, bobbing pelvis with IK made the arm servos resonate (hand bobbing 3-5 cm)
            # and the object crept out of the grip
            key = tuple(np.round(np.r_[self.wrist_pos, self.wrist_quat], 4))
            if self._carry_key != key:
                p_w, R_w = self.hand_target_world()
                self._carry_q, self.ik_err = self.ctrl.solve_best(self.d, p_w, R_w)
                self._carry_key = key
            self.ctrl.q_tgt = self.ctrl.q_tgt + np.clip(self._carry_q - self.ctrl.q_tgt, -self.ctrl.max_dq,
                                                        self.ctrl.max_dq)
        else:
            p_w, R_w = self.hand_target_world()
            self.ik_err = self.ctrl.ik(self.d, p_w, R_w)
        self.ctrl.apply(self.d, self.fingers)
        for _ in range(N_SUB):
            self.walker.physics_step()
            mujoco.mj_step(self.m, self.d)
        self.t += CTRL_DT
        self.pose = self._true_pose()
        a = CTRL_DT / 0.5                       # ~ one gait cycle
        if np.linalg.norm(self.pose[:2] - self.pose_f[:2]) > 0.3:      # teleported (test setups)
            self.pose_f = self.pose.copy()
        dyaw = (self.pose[2] - self.pose_f[2] + np.pi) % (2 * np.pi) - np.pi
        self.pose_f = self.pose_f + a * np.array([*(self.pose[:2] - self.pose_f[:2]), dyaw])
        self._contacts()
        self._arm_safety()
        nl = self.ctrl.left_arm_contacts(self.d)
        if nl and not self.stats["_in_larm"]:
            self.stats["left_arm_contacts"] += 1
        self.stats["_in_larm"] = bool(nl)
        if self.walker.fallen():
            self.stats["fell"] = True

    def _arm_swing(self):
        """Shoulder-pitch offsets (right, left arm vectors) that swing each arm with the opposite leg, as people
        walk: taken from the measured hip pitch around its gait average, so the swing follows the real stride and
        fades out when the legs stop (standing, base lock)."""
        q = self.d.qpos[self.hip_q]
        self.hip_mean += CTRL_DT / 0.8 * (q - self.hip_mean)
        dl, dr = ARM_SWING_GAIN * (q - self.hip_mean)
        if self.walker.mode not in ("walk", "holding"):
            dl = dr = 0.0
        sr, sl = np.zeros(7), np.zeros(7)
        sr[0] = np.clip(dl, -ARM_SWING_MAX, ARM_SWING_MAX)      # right arm with the left leg (G1: negative pitch
        sl[0] = np.clip(dr, -ARM_SWING_MAX, ARM_SWING_MAX)      # = forward for both hip and shoulder)
        sr[3] = 0.5 * max(0.0, -sr[0])                          # the elbow bends a little as the arm comes forward
        sl[3] = 0.5 * max(0.0, -sl[0])
        return sr, sl

    def _track(self, v_ref):
        """PI velocity tracking around the walking policy: on this model (Menagerie G1 + Inspire hands) the
        policy drifts ~9 cm/s sideways and under-tracks small commands; the integrator removes that bias."""
        a = CTRL_DT / 0.4
        self.vel_f = (1 - a) * self.vel_f + a * self.vel_body()       # gait-averaged measured velocity
        err = v_ref - self.vel_f
        self.vel_i = np.clip(self.vel_i + VEL_KI * err * CTRL_DT, -VEL_I_MAX, VEL_I_MAX)
        return v_ref + VEL_KP * err + self.vel_i

    def _robot_world_vel(self):
        """Gait-averaged base velocity in world (the pelvis sways +-5-10 cm/s while stepping in place)."""
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        v = self.vel_f
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

    def _approaching(self, other_xy):
        """True if the robot is moving towards `other_xy` (> 8 cm/s gait-averaged closing speed)."""
        rel = np.asarray(other_xy) - self.pose[:2]
        return float(np.dot(self._robot_world_vel(), rel) / (np.linalg.norm(rel) + 1e-6)) > 0.08

    def _contacts(self):
        """Contact events: robot vs furniture from MuJoCo contacts; robot vs person / pet on the footprint
        (they are mocap prisms without collision, so their 2D footprint separation is used, r = 0.25 m)."""
        touch_f = False
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            ba, bb = self.m.geom_bodyid[a], self.m.geom_bodyid[b]
            if (ba in self.robot_bodies and b in self.furniture_geoms) or (bb in self.robot_bodies and a in self.furniture_geoms):
                touch_f = True
                break
        if touch_f and not self.stats["_in_f"]:
            self.stats["furniture_contacts"] += 1
        self.stats["_in_f"] = touch_f
        for agent, key, kin, mind in ((self.person, "person_contacts", "_in_p", "min_person_dist"),
                                      (self.pet, "pet_contacts", "_in_pet", "min_pet_dist")):
            if agent is None:
                continue
            if agent is self.person:
                dist = np.linalg.norm(agent.pos - self.pose[:2]) - apt.PERSON_RADIUS - apt.ROBOT_RADIUS
            else:
                dist = obb_circle_distance(agent.pos, agent.yaw, apt.PET_HALF[:2], self.pose[:2]) - apt.ROBOT_RADIUS
            self.stats[mind] = min(self.stats[mind], dist)
            if dist <= 0.0:
                if not self.stats[kin]:
                    self.stats[key] += 1
                    if self._approaching(agent.pos):
                        self.stats[key.replace("contacts", "contacts_moving")] += 1
                self.stats[kin] = True
            else:
                self.stats[kin] = False

    def _arm_safety(self):
        """Runtime arm safety: count arm/hand contacts with furniture or the robot's own body; above
        CONTACT_STOP_N the arm stops where it is (targets = measured joints) instead of pushing on."""
        n, fmax = self.ctrl.arm_collisions(self.d)
        if n and not self.stats["_in_arm"]:
            self.stats["arm_contacts"] += 1
        self.stats["_in_arm"] = bool(n)
        if fmax > g1mod.CONTACT_STOP_N:
            self.stats["arm_stops"] += 1
            self.ctrl.q_tgt = self.d.qpos[self.ctrl.ik_q].copy()
            if self.arm_q_goal is not None:
                self.arm_q_goal = self.ctrl.q_tgt.copy()

    def pelvis_height(self):
        return float(self.d.xpos[self.robot_body][2])

    def observe(self):
        return {"pose": self.pose.copy(), "t": self.t}

    def fallen(self):
        return self.walker.fallen()
