"""Unitree G1 (29 DoF, MuJoCo Menagerie) + Inspire RH56 five-finger hands (Unitree URDF), in the apartment.

* Model: Menagerie `g1_with_hands.xml` with its Dex3 hands replaced by the Inspire DFQ hands from
  `unitree_ros/robots/g1_description/inspire_hand` (6 actuated DoF per hand, URDF <mimic> couplings as joint
  equalities). The left hand is visual only.
* Legs: the 12 leg actuators are torque motors driven by Unitree's pretrained walking policy
  (locomotion.walker.Walker, ported from the HomeHand project): base velocity commands (vx, vy, wz) in, balance
  stand for manipulation - the interface the plan uses for GEAR-SONIC.
* Right arm: damped-least-squares IK on the R_palm site, tracked by the arm's own position actuators.
* Right hand: 6 position targets; objects are held only through contact + friction.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
G1_DIR = ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1"
G1_XML = G1_DIR / "g1_with_hands.xml"
INSPIRE_DIR = ROOT / "third_party" / "unitree_ros" / "robots" / "g1_description"
PREFIX = ""                 # robot names kept as in the model (the Walker relies on them)
PELVIS_Z = 0.793            # Menagerie pelvis default height
JOINT_MARGIN = 0.10         # rad kept from every arm joint limit
ARM_MAX_VEL = 1.5           # rad/s, arm joint target speed limit
CONTACT_STOP_N = 25.0       # N: an arm / hand contact with furniture or the robot's own body above this stops the arm

WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
R_ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
         "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
L_ARM = [j.replace("right", "left") for j in R_ARM]
LEGS = [f"{s}_{j}_joint" for s in ("left", "right")
        for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
LEG_TORQUE = {"hip_pitch": 88, "hip_roll": 139, "hip_yaw": 88, "knee": 139, "ankle_pitch": 50, "ankle_roll": 50}
# Inspire actuated joints, order of the action vector: thumb yaw, thumb pitch, index, middle, ring, pinky
FINGERS = ["thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint", "index_proximal_joint",
           "middle_proximal_joint", "ring_proximal_joint", "pinky_proximal_joint"]
MIMIC = [("thumb_intermediate_joint", "thumb_proximal_pitch_joint", 1.6),
         ("thumb_distal_joint", "thumb_proximal_pitch_joint", 2.4),
         ("index_intermediate_joint", "index_proximal_joint", 1.0),
         ("middle_intermediate_joint", "middle_proximal_joint", 1.0),
         ("ring_intermediate_joint", "ring_proximal_joint", 1.0),
         ("pinky_intermediate_joint", "pinky_proximal_joint", 1.0)]
# Hand commands (radians). Grasp strategy taken from HomeHand's cylindrical power wrap: the thumb is turned
# into opposition *before* the approach, then thumb pitch + four fingers close.
HAND_OPEN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
HAND_PRESHAPE = np.array([1.3, 0.0, 0.0, 0.0, 0.0, 0.0])        # thumb yaw 1.3: grasp search on this model
HAND_CLOSED = np.array([1.3, 0.6, 1.47, 1.47, 1.47, 1.47])
RELAXED_HAND = np.array([0.0, 0.1, 0.3, 0.35, 0.4, 0.45])
# palm site in the right_wrist_yaw_link frame: between the thumb and the four fingers (cup-centre region found
# with the grasp search), x along the fingers, y = palm normal, z = finger spread (up in the handshake pose)
PALM_LOCAL = np.array([0.18, 0.0, 0.0])
# resting arm posture: upper arm slightly out from the hip, elbow bent, forearm forward (no hip / chair contact)
ARM_READY = {"shoulder_pitch_joint": -0.2, "shoulder_roll_joint": -0.3, "shoulder_yaw_joint": 0.0,
             "elbow_joint": 1.1, "wrist_roll_joint": 0.0, "wrist_pitch_joint": 0.0, "wrist_yaw_joint": 0.0}
# relaxed walking posture: arms hanging beside the body, elbows slightly bent (the arms swing from here, see
# HomeEnv._arm_swing); the right arm goes to the ready / carry pose only when the hand has work to do
ARM_REST = {"shoulder_pitch_joint": 0.15, "shoulder_roll_joint": -0.15, "shoulder_yaw_joint": 0.0,
            "elbow_joint": 0.45, "wrist_roll_joint": 0.0, "wrist_pitch_joint": 0.0, "wrist_yaw_joint": 0.0}
# idle (left) arm while the robot stands at the table / counter: upper arm back and out, elbow bent, hand beside
# the hip. From the rest posture the visual-only left hand reached ~9 cm into the counter at the sink
ARM_TUCK = {"shoulder_pitch_joint": 0.3, "shoulder_roll_joint": -0.2, "shoulder_yaw_joint": 0.0,
            "elbow_joint": 0.9, "wrist_roll_joint": 0.0, "wrist_pitch_joint": 0.0, "wrist_yaw_joint": 0.0}


def _load_inspire(side: str) -> mujoco.MjSpec:
    s = mujoco.MjSpec.from_file(str(INSPIRE_DIR / "inspire_hand" / f"DFQ_{side}_hand.urdf"))
    s.meshdir = str(INSPIRE_DIR / "meshes")
    for m in s.meshes:
        m.file = m.file.split("/")[-1]
        m.maxhullvert = 64
    return s


def _remove_dex3(g1: mujoco.MjSpec):
    for k in list(g1.keys):          # keyframe sizes change; the controller sets the pose itself
        g1.delete(k)
    for a in list(g1.actuators):
        if "_hand_" in a.name:
            g1.delete(a)
    for side in ("left", "right"):
        wrist = g1.body(f"{side}_wrist_yaw_link")
        for g in list(wrist.geoms):
            if g.meshname == f"{side}_hand_palm_link":
                g1.delete(g)
        for b in list(wrist.bodies):
            g1.delete(b)


def _attach_inspire(g1: mujoco.MjSpec, side: str):
    S = side[0].upper()
    hand = _load_inspire(side)
    wrist = g1.body(f"{side}_wrist_yaw_link")
    # the hand base link (with its fixed URDF offset) and all fingers hang below it
    wrist.add_frame().attach_body(hand.body(f"{S}_hand_base_link"), "", "")
    for b in g1.bodies:
        if b.name.startswith(f"{S}_"):
            b.gravcomp = 1.0
            if side == "left":     # left hand is not used for manipulation: visual only
                for g in list(b.geoms):
                    if g.contype:
                        g1.delete(g)
            for g in b.geoms:
                visual = g.contype == 0
                g.group = 2 if visual else 3
                if visual:
                    g.rgba = [0.75, 0.75, 0.78, 1]
                else:
                    g.friction = [1.5, 0.02, 0.002]
                    g.condim = 4
    for j in g1.joints:
        if j.name.startswith(f"{S}_"):
            j.damping = [0.05, 0, 0]
            j.armature = 0.001
            j.frictionloss = 0.0
    for follower, leader, k in MIMIC:
        eq = g1.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=f"{S}_{follower}", name2=f"{S}_{leader}",
                             objtype=mujoco.mjtObj.mjOBJ_JOINT)
        eq.data[:5] = [0, k, 0, 0, 0]
        eq.solref = [0.005, 1]
    for jn in FINGERS:
        j = g1.joint(f"{S}_{jn}")
        g1.add_actuator(name=f"{S}_{jn}", target=f"{S}_{jn}", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                        gaintype=mujoco.mjtGain.mjGAIN_FIXED, biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                        gainprm=[8.0] + [0] * 9, biasprm=[0, -8.0, -0.25] + [0] * 7,
                        ctrlrange=j.range, ctrllimited=True, forcerange=[-3, 3], forcelimited=True)


def _leg_motors(g1: mujoco.MjSpec):
    """Replace the Menagerie leg position servos by torque motors (the walking policy runs its own PD)."""
    for a in list(g1.actuators):
        if a.target in LEGS:
            g1.delete(a)
    for jn in LEGS:
        lim = next(v for k, v in LEG_TORQUE.items() if k in jn)
        g1.add_actuator(name=jn.replace("_joint", "_motor"), target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                        gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[1.0] + [0] * 9,
                        biastype=mujoco.mjtBias.mjBIAS_NONE, ctrlrange=[-lim, lim], ctrllimited=True)


def build_spec(world: mujoco.MjSpec) -> mujoco.MjSpec:
    """Attach G1 + Inspire hands to an apartment spec (in place) and return it."""
    g1 = mujoco.MjSpec.from_file(str(G1_XML))
    for m in g1.meshes:
        m.file = str(G1_DIR / "assets" / m.file)
        m.maxhullvert = 64
    g1.meshdir = ""
    _remove_dex3(g1)
    _attach_inspire(g1, "right")
    _attach_inspire(g1, "left")
    _leg_motors(g1)
    torso = g1.body("torso_link")
    torso.add_camera(name="head", pos=[0.10, 0, 0.44], xyaxes=[0, -1, 0, 0.62, 0, 0.78], fovy=75)
    wrist = g1.body("right_wrist_yaw_link")
    wrist.add_site(name="R_palm", pos=PALM_LOCAL, size=[0.008, 0, 0], rgba=[0, 1, 0, 0.6], group=5)
    wrist.add_camera(name="wrist", pos=[0.06, -0.06, 0.0], xyaxes=[0, 0, -1, -0.6, 0.8, 0], fovy=85)
    for b in g1.bodies:
        if b.name.startswith("right_") and any(w in b.name for w in ("shoulder", "elbow", "wrist")):
            b.gravcomp = 1.0
    world.option.timestep = 0.002
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world.worldbody.add_frame().attach_body(g1.body("pelvis"), PREFIX, "")
    eq = world.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name="base_lock", name1="pelvis",
                            objtype=mujoco.mjtObj.mjOBJ_BODY)
    eq.active = False
    eq.solref = [0.005, 1.0]        # stiff: the welded pelvis must not drift while the arm works
    return world


class G1Controller:
    """IK + actuator targets for waist, arms and hands (legs belong to the Walker)."""

    def __init__(self, m: mujoco.MjModel):
        self.m = m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, PREFIX + n)
        self.ik_joints = [jid(n) for n in R_ARM]
        assert min(self.ik_joints) >= 0
        self.ik_q = np.array([m.jnt_qposadr[j] for j in self.ik_joints])
        self.ik_v = np.array([m.jnt_dofadr[j] for j in self.ik_joints])
        # safety: the IK keeps every arm joint JOINT_MARGIN inside its mechanical limits
        self.lo = m.jnt_range[self.ik_joints, 0] + JOINT_MARGIN
        self.hi = m.jnt_range[self.ik_joints, 1] - JOINT_MARGIN
        self.act_of_joint = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
        self.ik_act = np.array([self.act_of_joint[j] for j in self.ik_joints])
        self.l_arm_j = [jid(n) for n in L_ARM]
        self.l_arm_act = np.array([self.act_of_joint[j] for j in self.l_arm_j])
        self.waist_act = np.array([self.act_of_joint[jid(n)] for n in WAIST])
        self.finger_j = [jid("R_" + n) for n in FINGERS]
        self.hand_act = np.array([self.act_of_joint[j] for j in self.finger_j])
        self.l_hand_act = np.array([self.act_of_joint[jid("L_" + n)] for n in FINGERS])
        self.f_lo = m.actuator_ctrlrange[self.hand_act, 0]
        self.f_hi = m.actuator_ctrlrange[self.hand_act, 1]
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, PREFIX + "R_palm")
        self.r_ready = np.array([ARM_READY[n.replace("right_", "")] for n in R_ARM])
        self.l_ready = np.array([ARM_READY[n.replace("left_", "")] * (-1 if "roll" in n else 1) for n in L_ARM])
        self.r_rest = np.array([ARM_REST[n.replace("right_", "")] for n in R_ARM])
        self.l_rest = np.array([ARM_REST[n.replace("left_", "")] * (-1 if "roll" in n else 1) for n in L_ARM])
        self.l_tuck = np.array([ARM_TUCK[n.replace("left_", "")] * (-1 if "roll" in n else 1) for n in L_ARM])
        self.l_tgt = self.l_rest.copy()          # left arm joint targets (rest posture + swing while walking)
        self.scratch = mujoco.MjData(m)
        self.q_tgt = self.r_ready.copy()
        pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "pelvis")
        arm_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "right_shoulder_pitch_link")

        def below(b, r):
            while b > 0:
                if b == r:
                    return True
                b = int(m.body_parentid[b])
            return False
        robot = {b for b in range(m.nbody) if below(b, pelvis)}
        self.arm_bodies = {b for b in robot if below(b, arm_root)}
        l_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "left_shoulder_pitch_link")
        self.arm_bodies_all = self.arm_bodies | {b for b in robot if below(b, l_root)}   # both arms + hands
        self.arm_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in self.arm_bodies and m.geom_contype[g]}
        l_bodies = {b for b in robot if below(b, l_root)}
        self.l_arm_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in l_bodies and m.geom_contype[g]}
        # what the arm must not touch: the robot's own torso / legs / other arm and the static scenery (world
        # body: walls, furniture); the manipulated object is allowed, the floor is never reached
        floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.forbidden = {g for g in range(m.ngeom)
                          if (m.geom_contype[g] or m.geom_conaffinity[g]) and g != floor
                          and ((m.geom_bodyid[g] in robot and m.geom_bodyid[g] not in self.arm_bodies)
                               or m.geom_bodyid[g] == 0)}
        self.max_dq = ARM_MAX_VEL * 0.02  # rad per 20 ms control step
        # natural working posture of the right arm (G1: negative shoulder pitch = arm forward)
        self.posture = np.array([-0.5, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.k_posture = 0.05

    def set_ready(self, d, relaxed=False):
        r = self.r_rest if relaxed else self.r_ready
        for j, v in zip(self.ik_joints, r):
            d.qpos[self.m.jnt_qposadr[j]] = v
        for j, v in zip(self.l_arm_j, self.l_rest):
            d.qpos[self.m.jnt_qposadr[j]] = v
        self.q_tgt = r.copy()
        self.l_tgt = self.l_rest.copy()

    def palm_fk(self, d, q):
        """Palm site pose (world) for right-arm joints `q` at the current robot state."""
        sd = self.scratch
        sd.qpos[:] = d.qpos
        sd.qpos[self.ik_q] = q
        mujoco.mj_kinematics(self.m, sd)
        return sd.site_xpos[self.site].copy(), sd.site_xmat[self.site].reshape(3, 3).copy()

    def left_arm_contacts(self, data):
        """Contacts of the left arm / hand with anything but itself (the floor excluded): monitoring only."""
        n = 0
        for i in range(data.ncon):
            c = data.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if (a in self.l_arm_geoms) != (b in self.l_arm_geoms) and (a in self.forbidden or b in self.forbidden
                                                                      or a in self.arm_geoms or b in self.arm_geoms):
                n += 1
        return n

    def _solve(self, d, q0, pos_w, R_w, iters, damp, w_rot):
        sd = self.scratch
        sd.qpos[:] = d.qpos
        sd.qpos[self.ik_q] = q0
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        e_p = np.zeros(3)
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, sd)
            mujoco.mj_comPos(self.m, sd)
            p = sd.site_xpos[self.site]
            R = sd.site_xmat[self.site].reshape(3, 3)
            e_p = pos_w - p
            e_r = 0.5 * (np.cross(R[:, 0], R_w[:, 0]) + np.cross(R[:, 1], R_w[:, 1]) + np.cross(R[:, 2], R_w[:, 2]))
            mujoco.mj_jacSite(self.m, sd, jp, jr, self.site)
            J = np.vstack([jp[:, self.ik_v], w_rot * jr[:, self.ik_v]])
            e = np.r_[e_p, w_rot * e_r]
            Jp = J.T @ np.linalg.inv(J @ J.T + damp ** 2 * np.eye(6))
            dq = Jp @ e
            # null space: drift towards a natural arm posture (shoulder forward, elbow bent, wrist straight)
            dq += (np.eye(len(dq)) - Jp @ J) @ (self.k_posture * (self.posture - sd.qpos[self.ik_q]))
            sd.qpos[self.ik_q] = np.clip(sd.qpos[self.ik_q] + np.clip(dq, -0.1, 0.1), self.lo, self.hi)
        mujoco.mj_kinematics(self.m, sd)
        err = float(np.linalg.norm(pos_w - sd.site_xpos[self.site]))
        return sd.qpos[self.ik_q].copy(), err

    def ik(self, d, pos_w, R_w, iters=6, damp=0.04, w_rot=0.3):
        """Update arm joint targets so the palm site reaches (pos_w, R_w) in world frame. Warm-started from the
        previous solution; if that branch got stuck (joint limits), re-solved from the ready posture."""
        q, err = self._solve(d, self.q_tgt, pos_w, R_w, iters, damp, w_rot)
        if err > 0.05:
            q2, err2 = self._solve(d, self.r_ready, pos_w, R_w, 40, damp, w_rot)
            if err2 < err - 0.02:
                q, err = q2, err2
        # joint speed limit of the arm targets (never jump to another IK branch at once)
        self.q_tgt = self.q_tgt + np.clip(q - self.q_tgt, -self.max_dq, self.max_dq)
        return err

    def arm_collisions(self, data):
        """(count, max normal force) of contacts between the arm/hand and forbidden geoms in `data`."""
        n, fmax = 0, 0.0
        f = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if (a in self.arm_geoms and b in self.forbidden) or (b in self.arm_geoms and a in self.forbidden):
                n += 1
                if data is not self.scratch:
                    mujoco.mj_contactForce(self.m, data, i, f)
                    fmax = max(fmax, abs(f[0]))
        return n, fmax

    def collides(self, d, q):
        """Collision check of an arm configuration before it is commanded (forward kinematics + MuJoCo collision
        detection on a scratch copy of the current state)."""
        sd = self.scratch
        sd.qpos[:] = d.qpos
        sd.qpos[self.ik_q] = q
        mujoco.mj_fwdPosition(self.m, sd)
        return self.arm_collisions(sd)[0] > 0

    def solve_best(self, d, pos_w, R_w, w_rot=0.3):
        """Full IK solve before a joint-space move: warm start, ready posture and a few perturbed starts."""
        rng = np.random.default_rng(0)
        starts = [self.q_tgt, self.r_ready] + [np.clip(self.r_ready + rng.normal(0, 0.5, 7), self.lo, self.hi)
                                                 for _ in range(4)]
        best = None
        for q0 in starts:
            q, err = self._solve(d, q0, pos_w, R_w, 60, 0.04, w_rot)
            cost = (err + 0.02 * np.linalg.norm(q - self.q_tgt)     # close to the current arm ...
                    + 0.02 * np.linalg.norm(q - self.posture)       # ... and to a natural posture
                    + (1.0 if self.collides(d, q) else 0.0))        # never a configuration in collision
            if best is None or cost < best[2]:
                best = (q, err, cost)
        return best[0], best[1]

    def apply(self, d, fingers):
        """fingers: 6 Inspire joint targets in radians (clipped to the actuator ranges)."""
        d.ctrl[self.ik_act] = self.q_tgt
        d.ctrl[self.l_arm_act] = self.l_tgt
        d.ctrl[self.waist_act] = 0.0
        d.ctrl[self.l_hand_act] = np.clip(RELAXED_HAND, self.m.actuator_ctrlrange[self.l_hand_act, 0],
                                          self.m.actuator_ctrlrange[self.l_hand_act, 1])
        d.ctrl[self.hand_act] = np.clip(fingers, self.f_lo, self.f_hi)

    def site_pose(self, d):
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()
