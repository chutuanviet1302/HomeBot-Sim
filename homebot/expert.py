"""Scripted expert for pick / place with the G1 + Inspire hand (demo generator and fallback policy).

Grasp strategy (taken from the HomeHand project's cylindrical power wrap, re-measured on this model with
scripts/grasp_search.py): "handshake" palm orientation (fingers forward, palm facing left, thumb up), the thumb
turned into opposition *before* the approach, the hand slides forward along the finger direction so the object
enters between thumb and fingers without being swept, then thumb + fingers close and the arm lifts.
The robot stands still while the hand works (pose hold, then base lock - see HomeEnv.step); palm targets are fixed in the world so the arm
compensates the slow sway of the standing robot. The expert sees the object pose with perception-like noise
(sigma = 8 mm); whether it holds the object is decided by the contacts alone.
"""
from __future__ import annotations

import numpy as np

from . import apartment as apt
from . import g1 as g1mod
from .env import CARRY_DEFAULT, CARRY_HIGH, CTRL_DT, mat_to_quat, quat_to_mat, yaw_quat

# object centre relative to the palm site in the closed handshake grasp (hand frame: x fingers, y palm normal,
# z thumb side): centre of the most robust region found by scripts/grasp_search.py (results/grasp_search.csv):
# 1-3 cm behind the knuckles, object radius + 3-4 cm off the palm plane (room for the open fingers), palm 1-2 cm
# above the object centre (the little finger must clear the table)
GRASP_DX, GRASP_GAP, GRASP_DZ = -0.01, 0.01, -0.02
OBJ_IN_PALM = {k: np.array([GRASP_DX, v["r"] + GRASP_GAP, GRASP_DZ]) for k, v in apt.CUP_TYPES.items()}
APPROACH_BACKOFF = 0.13     # the palm site (at the knuckles) starts this far behind the grasp pose: the open
                            # fingers reach ~5-8 cm beyond it and must not sweep the object (HomeHand: 7 cm from
                            # the palm centre)
PRE_HIGH = 0.10             # the hand comes down behind the object from this high above the grasp height
OBJ_IN_BASE = np.array([0.32, -0.20])   # where the object should be in the pelvis frame when the hand starts
                            # (the stance of the nominal successful grasp; ALIGN walks there first)
DROP_IN_BASE = np.array([0.45, -0.20])  # drop spot in the pelvis frame: the counter keeps the robot 30 cm away
ALIGN_TOL = 0.04
FINE_SPEED = 0.15           # m/s, approach / place
FREE_SPEED = 0.35           # m/s, free-space moves
LIFT = 0.12
LIFT_SPEED = 0.08           # m/s
THUMB_YAW_APPROACH = 1.3   # thumb rotation while the hand slides in (1.3 = already in opposition)
OBJ_NOISE = 0.008           # m, horizontal perception noise of the object position
OBJ_NOISE_Z = 0.002         # m, height: the object rests on a support plane of known height
CARRY = CARRY_DEFAULT


def to_robot(pose, p_world):
    x, y, th = pose
    d = np.asarray(p_world)[:2] - np.array([x, y])
    c, s = np.cos(th), np.sin(th)
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], p_world[2]])


def rot6d_mat(R):
    return np.asarray(R)[:, :2].T.reshape(-1)


def rot6d(q):
    return rot6d_mat(quat_to_mat(q))


def min_jerk(s):
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def slerp(q0, q1, s):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)
    return q / np.linalg.norm(q)


class Expert:
    def __init__(self, env, recorder=None, rng=None):
        self.env = env
        self.rec = recorder
        self.rng = rng or np.random.default_rng(0)

    # -------------------------------------------------------------- low level
    def _palm_target_robot(self):
        """Current palm target in the pelvis frame (position, rotation)."""
        e = self.env
        if e.arm_q_goal is not None or (e.arm_relaxed and e.hand_goal_world is None):
            p_w, R_w = e.ctrl.palm_fk(e.d, e.ctrl.q_tgt)       # joint-space motion: the commanded arm pose
        else:
            p_w, R_w = e.hand_target_world()
        Rb = quat_to_mat(yaw_quat(e.pose[2]))
        return e.world_to_robot(p_w), Rb.T @ R_w

    def _action(self):
        """17-D action: palm pos (pelvis frame) | palm rot6d | pelvis height, stand flag | 6 finger targets."""
        e = self.env
        p, R = self._palm_target_robot()
        return np.concatenate([p - np.array([0, 0, g1mod.PELVIS_Z]), rot6d_mat(R),
                               [e.pelvis_height(), 1.0 if e.stand_mode else 0.0], e.fingers])

    def _tick(self):
        if self.rec is not None:
            self.rec.add(self.env, self._action())
        self.env.step(np.zeros(3), carrying=self.env.grasped)

    def align(self, obj_w, rel=OBJ_IN_BASE, timeout=12.0, refine=None):
        """Step (walking policy) until the perceived object sits at `rel` in the pelvis frame, facing it.
        `refine()` (optional) returns an updated object estimate; it is called every 0.5 s while stepping."""
        e = self.env
        t_ref = e.t
        e.stand_mode = False
        e.hand_goal_world = None
        t0 = e.t
        ok_n = 0
        err_f = None                                     # low-pass of the stance error (the pose estimate is noisy)
        while e.t - t0 < timeout:
            if refine is not None and e.t - t_ref >= 0.5:
                t_ref = e.t
                new = refine()
                if new is not None:
                    obj_w = new
            x, y, th = e.est_pose()              # control on the current estimate (a lagging average destabilises
                                                 # the loop); convergence is judged on the low-passed error below
            # heading such that the object is seen under the angle of `rel`, then position
            d = np.asarray(obj_w[:2]) - [x, y]
            yaw_t = np.arctan2(d[1], d[0]) - np.arctan2(rel[1], rel[0])
            c, s_ = np.cos(yaw_t), np.sin(yaw_t)
            base_t = np.asarray(obj_w[:2]) - np.array([c * rel[0] - s_ * rel[1], s_ * rel[0] + c * rel[1]])
            e_w = base_t - [x, y]
            ex_, ey_ = np.cos(th) * e_w[0] + np.sin(th) * e_w[1], -np.sin(th) * e_w[0] + np.cos(th) * e_w[1]
            eyaw = (yaw_t - th + np.pi) % (2 * np.pi) - np.pi
            err = np.array([ex_, ey_, eyaw])
            err_f = err if err_f is None else err_f + CTRL_DT / 0.5 * (err - err_f)
            if np.hypot(err_f[0], err_f[1]) < ALIGN_TOL and abs(err_f[2]) < 0.05:
                ok_n += 1
                if ok_n > 10:
                    return True
            else:
                ok_n = 0
            vmax = 0.12 if e.grasped else 0.2          # small, gentle steps while holding the object
            cmd = np.array([np.clip(1.2 * ex_, -vmax, vmax), np.clip(1.2 * ey_, -0.12, 0.12),
                            np.clip(1.5 * eyaw, -0.4, 0.4)])
            if self.rec is not None:
                self.rec.add(e, self._action())
            e.step(cmd, carrying=e.grasped)
        return False

    def align_and_stand(self, rel, target_w=None, tries=4, tol=0.07):
        """ALIGN on the perceived object, then stand still; if the stance drifted away while locking (> `tol`),
        step again. Returns True once standing with the object where the hand works."""
        e = self.env
        self.ready_arm()                                   # hand up before stepping close to the furniture
        for _ in range(tries):
            e.stats["align_tries"] = e.stats.get("align_tries", 0) + 1
            if target_w is not None:                       # a fixed spot of the map (the sink)
                obj, refine = np.asarray(target_w), None
            else:                                          # the object: perception, refined while stepping
                obj = self.perceive_object()
                if obj is None:                            # not seen from here: the map's usual spot
                    obj = np.array([*apt.CUP_HOME, apt.TABLE_TOP + e.cup_h / 2])
                refine = self.perceive_object
            self.align(obj, rel, refine=refine)
            if not self.stand():
                continue
            if refine is not None:
                obj = self.perceive_object(frames=3, retries=3)
                if obj is None:
                    e.stats["not_seen"] = e.stats.get("not_seen", 0) + 1
                    if e.perception is not None:
                        e.stats["not_seen_info"] = str(e.perception.last_clusters)[:200]
                    e.stand_mode = False
                    e.hand_goal_world = None
                    continue
            if refine is not None:
                if self.reachable(obj):                # the object: every grasp pose reachable from here
                    return True
                e.stats["unreachable"] = e.stats.get("unreachable", 0) + 1
            elif np.linalg.norm(e.world_to_robot(obj)[:2] - rel) < tol:
                return True
            e.stand_mode = False
            e.hand_goal_world = None
        return False                                   # never work with the object out of the hand's reach

    def stand(self, max_t=4.0):
        """Stand still for the hands (hold, then base lock); from then on the palm target is held in the world."""
        e = self.env
        e.stand_mode = True
        t0 = e.t
        while e.walker.mode not in ("stand", "locked") and e.t - t0 < max_t:
            self._tick()
        standing = e.walker.mode in ("stand", "locked")
        if e.arm_relaxed:
            if not standing:
                return False
            # the hand gets ready only now, standing still: collision-checked joint path from the hanging arm
            self.wait(0.3)
            self.move_joint(*e.robot_to_world(e.wrist_pos, e.wrist_quat), speed=0.8)
            e.arm_relaxed = False
        if e.hand_goal_world is None:
            e.hand_goal_world = e.robot_to_world(e.wrist_pos, e.wrist_quat)
        self.wait(0.3)
        return standing

    def move_hand(self, target_w, R_w, speed=FREE_SPEED, fingers=None):
        """Minimum-jerk palm move to a world pose (fingers blend to `fingers` on the way)."""
        e = self.env
        p0, R0 = e.hand_goal_world
        q0, q1 = mat_to_quat(R0), mat_to_quat(R_w)
        f0 = e.fingers.copy()
        f1 = f0 if fingers is None else np.asarray(fingers, float)
        n = max(5, int(np.linalg.norm(np.asarray(target_w) - p0) / (speed * CTRL_DT) * 1.6))
        for i in range(1, n + 1):
            s = min_jerk(i / n)
            e.hand_goal_world = (p0 + s * (np.asarray(target_w) - p0), quat_to_mat(slerp(q0, q1, s)))
            e.fingers = f0 + s * (f1 - f0)
            self._tick()

    def move_joint(self, target_w, R_w, speed=1.0, fingers=None):
        """Free-space move: IK solved once for the goal, the arm moves in joint space (min-jerk), then the palm
        target is set to the goal so the following Cartesian moves start from it."""
        e = self.env
        q_goal, err = e.ctrl.solve_best(e.d, np.asarray(target_w), R_w)
        q0 = e.ctrl.q_tgt.copy()
        f0 = e.fingers.copy()
        f1 = f0 if fingers is None else np.asarray(fingers, float)
        # safety: the joint-space path is collision-checked (furniture and the robot's own body) before the arm
        # moves; if it would touch something, a two-segment path through a via configuration is searched
        # (over the goal, over the start, pulled back, halfway and raised, the ready posture). Only a fully
        # collision-free path is executed; if none exists the arm moves slowly and the contact stop guards it.
        free = lambda qa, qb: not any(e.ctrl.collides(e.d, qa + t * (qb - qa)) for t in np.linspace(0, 1, 13)[1:])
        if free(q0, q_goal):
            self._joint_path(q0, q_goal, speed, f0, f1)
        else:
            e.stats["arm_paths_rejected"] += 1
            p0, _ = e.hand_world()
            tgt = np.asarray(target_w, float)
            back = np.asarray(R_w)[:, 0] * np.array([1, 1, 0])
            back = -back / (np.linalg.norm(back) + 1e-9)
            vias = [e.ctrl.solve_best(e.d, w, R_w)[0] for w in
                    (tgt + [0, 0, 0.10], p0 + [0, 0, 0.10], p0 + 0.12 * back + [0, 0, 0.08],
                     0.5 * (p0 + tgt) + [0, 0, 0.15])] + [e.ctrl.r_ready.copy()]
            for via in vias:
                if free(q0, via) and free(via, q_goal):
                    self._joint_path(q0, via, speed, f0, f0)
                    self._joint_path(e.ctrl.q_tgt.copy(), q_goal, speed, f0, f1)
                    break
            else:
                e.stats["arm_paths_blocked"] = e.stats.get("arm_paths_blocked", 0) + 1
                self._joint_path(q0, q_goal, 0.3 * speed, f0, f1)
        e.arm_q_goal = None
        e.hand_goal_world = (np.asarray(target_w, float), np.asarray(R_w, float))
        return err

    def _joint_path(self, q0, q1, speed, f0, f1):
        e = self.env
        n = max(10, int(np.max(np.abs(q1 - q0)) / (speed * CTRL_DT) * 1.8))
        for i in range(1, n + 1):
            s_ = min_jerk(i / n)
            e.arm_q_goal = q0 + s_ * (q1 - q0)
            e.fingers = f0 + s_ * (f1 - f0)
            self._tick()
        for _ in range(10):
            self._tick()

    def hand(self, value, dur=0.5):
        e = self.env
        f0 = e.fingers.copy()
        n = max(1, int(dur / CTRL_DT))
        for i in range(1, n + 1):
            e.fingers = f0 + (np.asarray(value, float) - f0) * min_jerk(i / n)
            self._tick()

    def wait(self, dur):
        for _ in range(int(dur / CTRL_DT)):
            self._tick()

    def to_carry(self, holding=False):
        """Back to the rest / carry pose (robot frame) and hand the arm back to the walking controller."""
        e = self.env
        pose = CARRY_HIGH if holding else CARRY
        p, R = e.robot_to_world(pose)
        self.move_joint(p, R, speed=0.4 if holding else 0.8)     # gentle with an object in the hand
        e.hand_goal_world = None
        e.wrist_pos, e.wrist_quat = pose.copy(), np.array([1.0, 0, 0, 0])
        if not holding and e.relaxed_arms:
            self.relax_arm()
        e.stand_mode = False

    def ready_arm(self):
        """Empty hand arriving at the work place: raise the hanging arm to the ready pose (collision-checked joint
        path, stepping in place) before stepping close - a hanging hand would brush the table / counter edge."""
        e = self.env
        if not e.arm_relaxed:
            return
        e.stand_mode = False
        self.move_joint(*e.robot_to_world(e.wrist_pos, e.wrist_quat), speed=0.8)
        e.hand_goal_world = None                         # walking again: the palm target follows the robot
        e.arm_relaxed = False

    def relax_arm(self):
        """Empty hand: lower the arm to the hanging walking posture (collision-checked joint path)."""
        e = self.env
        q0, q1 = e.ctrl.q_tgt.copy(), e.ctrl.r_rest.copy()
        if any(e.ctrl.collides(e.d, q0 + t * (q1 - q0)) for t in np.linspace(0, 1, 13)[1:]):
            e.stats["arm_paths_rejected"] += 1
            return                                   # stay in the carry pose (collision-free) rather than risk it
        self._joint_path(q0, q1, 0.8, e.fingers.copy(), e.fingers.copy())
        e.arm_q_goal = None
        e.arm_relaxed = True

    def perceive_object(self, frames=1, rotation=False, retries=0):
        """Object pose estimate from the head camera (detection + recognition + ICP 6D pose + tracking).
        Without perception (env.use_perception=False): the simulator pose with perception-like noise."""
        e = self.env
        if e.perception is None:
            cp, cq = e.cup_pose()
            p = cp + self.rng.normal(0, [OBJ_NOISE, OBJ_NOISE, OBJ_NOISE_Z])
            return (p, quat_to_mat(cq)) if rotation else p
        est = e.perception.locate(e.cup_type, frames=frames)
        for _ in range(retries):                       # standing still: look again a little later
            if est is not None:
                break
            self.wait(0.3)
            est = e.perception.locate(e.cup_type, frames=frames, max_misses=2)
        if est is None:
            return None
        return est if rotation else est[0]

    def object_in_hand(self):
        """Pose (centre, rotation) of the held object in the world: perception estimate (6D, head camera) when
        available, otherwise the simulator pose with perception-like noise."""
        e = self.env
        if e.perception is not None and e.held_rel is not None:
            # the pose measured by the camera just before the hand closed, carried along by the hand kinematics
            hp, hq = e.hand_world()
            Rh = quat_to_mat(hq)
            return hp + Rh @ e.held_rel[0], Rh @ e.held_rel[1]
        cp, cq = e.cup_pose()
        noise_R = upright_correction(np.r_[self.rng.normal(0, 0.03, 2), 1.0]).T
        return cp + self.rng.normal(0, OBJ_NOISE / 2, 3), noise_R @ quat_to_mat(cq)

    # -------------------------------------------------------------- skills
    def pick_cup(self):
        e = self.env
        if not self.align_and_stand(OBJ_IN_BASE):
            e.stand_mode = False
            return False
        return self.grasp_object()

    def _grasp_frames(self, est):
        e = self.env
        R = quat_to_mat(yaw_quat(e.pose[2]))           # handshake: fingers along the robot's heading
        return R, est - R @ OBJ_IN_PALM[e.cup_type], R @ np.array([-APPROACH_BACKOFF, 0, 0])

    def reachable(self, est, tol=0.01):
        """Can the hand reach every pose of the grasp from the current stance (IK error < tol, collision-free)?"""
        e = self.env
        R, grasp, back = self._grasp_frames(est)
        for p in (grasp + back + [0, 0, PRE_HIGH], grasp + back, grasp, grasp + [0, 0, LIFT]):
            q, err = e.ctrl.solve_best(e.d, p, R)
            if err > tol or e.ctrl.collides(e.d, q):
                return False
        return True

    def grasp_object(self, carry=True):
        """The grasp itself, robot already standing still in front of the object."""
        e = self.env
        meas = self.perceive_object(frames=3, rotation=True, retries=3)
        if meas is None:
            return False
        est, R_obj = meas
        e.stats["perc_err_mm"] = round(1000 * float(np.linalg.norm(est - e.cup_pose()[0])), 1)   # evaluation only
        R, grasp, back = self._grasp_frames(est)
        # reach behind the object (thumb into opposition on the way), come down, then slide forward
        pre = g1mod.HAND_PRESHAPE.copy()
        pre[0] = THUMB_YAW_APPROACH
        self.move_joint(grasp + back + [0, 0, PRE_HIGH], R, fingers=pre)
        self.move_hand(grasp + back, R, speed=FINE_SPEED)
        self.move_hand(grasp, R, speed=FINE_SPEED)
        if THUMB_YAW_APPROACH != g1mod.HAND_PRESHAPE[0]:
            self.hand(g1mod.HAND_PRESHAPE, 0.4)        # thumb into opposition once the object is in the hand
        fingers_first = g1mod.HAND_CLOSED.copy()
        fingers_first[1] = 0.0                       # four fingers wrap first, the thumb pitch closes last
        # grasp -> squeeze (as in SIMPLE / BODex): fingers close onto the object, then the targets stay past the
        # contact for a moment so the position servos build the grip force before anything moves
        # object pose in the hand frame at closing (camera estimate, the object is still standing on the table)
        hp, hq = e.hand_world()
        Rh = quat_to_mat(hq)
        e.held_rel = (Rh.T @ (est - hp), Rh.T @ R_obj)
        self.hand(fingers_first, 0.4)
        self.hand(g1mod.HAND_CLOSED, 0.4)
        self.wait(0.3)
        # slow lift: the grip settles while the object leaves the table
        self.move_hand(e.hand_goal_world[0] + [0, 0, LIFT], R, speed=LIFT_SPEED)
        self.wait(0.3)
        if not carry:
            cp, _ = e.cup_pose()
            return bool(e.grasped and cp[2] > apt.TABLE_TOP + e.cup_h / 2 + 0.05)
        self.to_carry(holding=True)
        self.wait(0.3)
        cp, _ = e.cup_pose()
        return bool(e.grasped and cp[2] > apt.TABLE_TOP + e.cup_h / 2 + 0.05)

    def place_cup(self, target_world=None):
        """Drop the held object into the sink: ALIGN on the basin, hand over the opening (object bottom a few cm
        above the rim), open the hand, retract."""
        e = self.env
        if target_world is None:
            # front half of the basin (the robot cannot lean over the counter), object bottom 4 cm above the rim
            target_world = np.array([*(apt.SINK_CENTER + [0, -0.03] + self.rng.uniform(-0.03, 0.03, 2)),
                                     apt.COUNTER_TOP + e.cup_h / 2 + 0.04])
        if not e.grasped:
            return False
        if not self.align_and_stand(DROP_IN_BASE, target_world) or not e.grasped:
            e.stand_mode = False
            return False
        # where the object sits in the closed hand (6D estimate), so that the object - not the palm - ends up
        # over the basin
        hp, hq = e.hand_world()
        Rh = quat_to_mat(hq)
        cp, _ = self.object_in_hand()
        held = Rh.T @ (cp - hp)
        R = quat_to_mat(yaw_quat(e.pose[2]))
        palm = target_world - R @ held
        self.move_joint(palm + [0, 0, 0.08], R)          # collision-checked: over the counter edge
        self.move_hand(palm, R, speed=0.10)
        self.hand(g1mod.RELAXED_HAND, 0.4)              # open: the object falls into the basin
        self.wait(0.6)
        self.move_hand(e.hand_goal_world[0] + [0, 0, 0.06], R, speed=0.2)
        self.to_carry()
        self.wait(0.4)
        return place_success(e)


def upright_correction(a, max_angle=np.deg2rad(50)):
    """Smallest rotation taking the world direction `a` (object axis) to +z, limited to `max_angle`."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    if a[2] < 0:
        a = -a
    ax = np.cross(a, [0, 0, 1.0])
    s = np.linalg.norm(ax)
    if s < 1e-6:
        return np.eye(3)
    ang = min(np.arctan2(s, a[2]), max_angle)
    k = ax / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def place_success(env):
    """The object lies inside the sink basin (below the rim, within the opening) and the hand let go."""
    cp, _ = env.cup_pose()
    in_xy = np.all(np.abs(cp[:2] - apt.SINK_CENTER) <= apt.SINK_HALF)
    in_basin = cp[2] < apt.COUNTER_TOP
    thumb, other, _ = env.cup_contacts()
    return bool(in_xy and in_basin and not (thumb or other))
