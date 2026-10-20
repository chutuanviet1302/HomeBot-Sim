"""Apartment layout (~8 x 6 m) and MJCF generation.

Coordinates: x in [0, 8], y in [0, 6]. Living room x < 4.5, kitchen/dining x > 4.5,
connected by a 1.0 m doorway in the partition wall (y in [3.8, 4.8]).
Passages are kept >= 0.9 m (G1 footprint 2 x 0.37 m + clearance), e.g. 1.2 m between the dining table (top
edge y=4.20) and the kitchen counter (bottom edge y=5.40), 1.35 m between the partition and the table.
"""
from __future__ import annotations

import copy
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

WIDTH, HEIGHT = 8.0, 6.0
GRID_RES = 0.05

CLEARANCE = 0.05        # m kept between the footprint and obstacles (DWA hard margin)
PLAN_MARGIN = 0.04      # m more for the global planner: gaps right at the DWA limit are not planned through
LAYOUT_SLACK = 0.0      # m: extra width required next to displaced chairs (the planner compensates the LiDAR map's
                        # cell quantisation itself, see Navigator._global_map)
ROBOT_RADIUS = 0.37     # G1 footprint while walking: right Inspire hand in the carry pose reaches 0.37 m from
                        # the pelvis axis (geometry sampled over walking / turning / side-stepping)
ROBOT_HEIGHT = 1.32
PERSON_RADIUS = 0.20    # cylinder D40 cm, 170 cm tall
PERSON_HEIGHT = 1.70
PET_HALF = (0.25, 0.125, 0.15)  # box 50 x 25 x 30 cm

TABLE_TOP = 0.75
COUNTER_TOP = 0.90


@dataclass
class Item:
    name: str
    kind: str                 # "box" | "cyl"
    pos: tuple                # (x, y)
    size: tuple               # (hx, hy) for box, (r,) for cyl
    h: float                  # height
    rgba: str = "0.6 0.6 0.6 1"
    movable: bool = False     # randomly displaced each episode (not on the map)
    yaw: float = 0.0


def base_layout() -> list[Item]:
    wall = "0.85 0.83 0.78 1"
    wood = "0.55 0.38 0.22 1"
    fabric = "0.30 0.40 0.55 1"
    chair = "0.45 0.30 0.18 1"
    items = [
        # outer walls
        Item("wall_s", "box", (4.0, -0.05), (4.1, 0.05), 2.2, wall),
        Item("wall_n", "box", (4.0, 6.05), (4.1, 0.05), 2.2, wall),
        Item("wall_w", "box", (-0.05, 3.0), (0.05, 3.1), 2.2, wall),
        Item("wall_e", "box", (8.05, 3.0), (0.05, 3.1), 2.2, wall),
        # partition with doorway y in [3.8, 4.8]
        Item("wall_part_s", "box", (4.5, 1.9), (0.05, 1.9), 2.2, wall),
        Item("wall_part_n", "box", (4.5, 5.4), (0.05, 0.6), 2.2, wall),
        # living room
        Item("sofa", "box", (1.3, 0.55), (1.0, 0.45), 0.80, fabric),
        Item("coffee_table", "box", (1.3, 1.80), (0.5, 0.30), 0.45, wood),
        Item("tv_stand", "box", (0.25, 4.5), (0.20, 0.75), 0.60, "0.2 0.2 0.2 1"),
        Item("bookshelf", "box", (3.5, 5.75), (0.50, 0.20), 1.80, wood),
        Item("toy_box", "box", (2.6, 5.25), (0.25, 0.2), 0.30, "0.8 0.6 0.2 1"),
        Item("swivel_chair", "cyl", (3.2, 1.4), (0.30,), 0.90, "0.15 0.15 0.15 1", movable=True),
        # kitchen / dining
        Item("counter", "box", (6.8, 5.7), (1.2, 0.30), COUNTER_TOP, "0.75 0.75 0.78 1"),
        # 1.2 x 0.8 m table; chairs leave >= 0.9 m walkways (west of the table: doorway -> south side)
        Item("dining_table", "box", (6.5, 3.80), (0.60, 0.40), TABLE_TOP, wood),
        Item("chair_2", "box", (7.35, 3.80), (0.225, 0.225), 0.90, chair, movable=True),
        Item("chair_3", "box", (5.75, 2.55), (0.225, 0.225), 0.90, chair, movable=True),
        Item("chair_4", "box", (7.25, 2.55), (0.225, 0.225), 0.90, chair, movable=True),
        Item("fridge", "box", (7.6, 0.5), (0.35, 0.35), 1.80, "0.9 0.9 0.92 1"),
    ]
    return items


# Named places used by tasks. "align" = where the robot stands (35 cm from the edge
# of the robot footprint to the furniture edge measured at the robot centre), facing yaw.
PLACES = {
    "living_room": {"pos": (2.2, 3.0), "yaw": 0.0},
    "kitchen": {"pos": (6.5, 1.5), "yaw": 0.0},
    # cup sits near the south edge of the dining table
    "table": {"pos": (6.5, 3.10), "yaw": np.pi / 2, "pre": (6.5, 2.25)},
    # sink is on the counter at x=7.0
    "sink": {"pos": (7.0, 5.05), "yaw": np.pi / 2, "pre": (7.45, 4.85)},
    "toy_box": {"pos": (2.6, 4.9), "yaw": np.pi / 2, "pre": (2.6, 4.3)},
}
CUP_HOME = np.array([6.5, 3.52])           # nominal cup xy on the table
SINK_CENTER = np.array([7.0, 5.60])         # sink basin centre (recessed into the counter, near its front edge)
SINK_HALF = np.array([0.20, 0.13])          # inner half size of the basin opening
SINK_DEPTH = 0.18
# blind spots: people can step out from behind the partition here, the LiDAR sees them only ~0.8 m away
DOORWAYS = [np.array([4.5, 4.3])]
BLIND_SPEED, BLIND_RADIUS = 0.2, 1.6      # the gait brakes at ~0.8 m/s^2: be slow before the corner


def randomize_layout(rng: np.random.Generator, displace=0.30) -> list[Item]:
    """Displace movable chairs by up to +-displace (m), keeping key spots reachable."""
    base = base_layout()
    for _ in range(200):
        items = copy.deepcopy(base)
        for it in items:
            if it.movable:
                dx, dy = rng.uniform(-displace, displace, 2)
                it.pos = (it.pos[0] + dx, it.pos[1] + dy)
                it.yaw = rng.uniform(-0.4, 0.4) if it.kind == "box" else 0.0
        if _layout_ok(items):
            return items
    return base


def _layout_ok(items: list[Item]) -> bool:
    occ = rasterize(items, inflate=ROBOT_RADIUS + 0.02)
    for name in ("table", "sink"):
        p = PLACES[name]
        # the straight approach corridor pre -> align must be free (its last part, next to the table / counter
        # edge, is walked by ALIGN at close range)
        a, b = np.array(p["pre"]), np.array(p["pos"])
        for t in np.linspace(0, 0.7, 11):
            if occupied(occ, a + t * (b - a)):
                return False
    # manipulation workspace: no chair within reach of the arms (+ the E2 start perturbation) around the
    # stances where the robot picks / drops - a person tidying the room would leave that space free too
    occ_mov = rasterize([i for i in items if i.movable], inflate=0.0)
    for name in ("table", "sink"):
        c = np.array(PLACES[name]["pos"])
        for r in (0.2, 0.4, 0.55):                  # arm reach + start perturbation
            for a in np.linspace(-np.pi, np.pi, 24, endpoint=False):
                if occupied(occ_mov, c + r * np.array([np.cos(a), np.sin(a)])):
                    return False
    # chairs must not overlap the table, each other, or walls (rough check)
    occ0 = rasterize([i for i in items if not i.movable], inflate=0.0)
    for it in items:
        if it.movable and occupied(occ0, np.array(it.pos)):
            return False
    for k in ("living_room", "kitchen"):
        if occupied(occ, np.array(PLACES[k]["pos"])):
            return False
    # all rooms must stay connected for the planner (inflated like the robot's A*)
    from scipy.ndimage import label
    from .nav.astar import inflate as _infl
    r_plan = ROBOT_RADIUS + CLEARANCE + PLAN_MARGIN
    blocked = (_infl(rasterize([i for i in items if not i.movable]), r_plan)
               | _infl(rasterize([i for i in items if i.movable]), r_plan + LAYOUT_SLACK))
    lab, _ = label(~blocked)                                  # passable for the planners
    ids = {int(lab[world_to_cell(np.array(PLACES[k].get("pre", PLACES[k]["pos"])))])
           for k in ("living_room", "kitchen", "table", "sink")}
    return len(ids) == 1 and 0 not in ids


# ----------------------------------------------------------------------------- grid
def grid_shape():
    return int(round(HEIGHT / GRID_RES)), int(round(WIDTH / GRID_RES))


def world_to_cell(p):
    p = np.asarray(p)
    return np.floor(p[..., 1] / GRID_RES).astype(int), np.floor(p[..., 0] / GRID_RES).astype(int)


def cell_to_world(r, c):
    return np.stack([(np.asarray(c) + 0.5) * GRID_RES, (np.asarray(r) + 0.5) * GRID_RES], -1)


def occupied(occ, p) -> bool:
    r, c = world_to_cell(p)
    if r < 0 or c < 0 or r >= occ.shape[0] or c >= occ.shape[1]:
        return True
    return bool(occ[r, c])


def rasterize(items: list[Item], inflate=0.0) -> np.ndarray:
    """2D occupancy (rows=y, cols=x) of item footprints, inflated by `inflate` m."""
    H, W = grid_shape()
    centers = cell_to_world(*np.meshgrid(np.arange(H), np.arange(W), indexing="ij"))
    occ = np.zeros((H, W), bool)
    for it in items:
        d = centers - np.array(it.pos)
        if it.kind == "cyl":
            occ |= np.linalg.norm(d, axis=-1) <= it.size[0] + inflate
        else:
            c, s = np.cos(-it.yaw), np.sin(-it.yaw)
            lx = c * d[..., 0] - s * d[..., 1]
            ly = s * d[..., 0] + c * d[..., 1]
            qx = np.maximum(np.abs(lx) - it.size[0], 0)
            qy = np.maximum(np.abs(ly) - it.size[1], 0)
            occ |= np.hypot(qx, qy) <= inflate + 1e-9
    # outer walls lie just outside the grid: mark the border so dilation sees them
    occ[0, :] = occ[-1, :] = True
    occ[:, 0] = occ[:, -1] = True
    return occ


def static_map(inflate=0.0) -> np.ndarray:
    """The known map: walls and fixed furniture only. Chairs get moved around every day,
    so they are not on the map - the robot must discover them with its LiDAR."""
    return rasterize([i for i in base_layout() if not i.movable], inflate)


# ----------------------------------------------------------------------------- MJCF
YCB_DIR = Path(__file__).resolve().parents[1] / "third_party" / "ycb"
# YCB objects (google_16k meshes). r / h = body radius / height, centre = body axis in the mesh frame (the mesh
# origin is on the table under the object), mass from the YCB object list.
CUP_TYPES = {
    # the YCB tomato soup can: the classic power-grasp object. (Dropped: the small 13 g YCB cup 065-a slipped out
    # of the Inspire hand, and the YCB mug 025_mug failed whenever its handle faced the fingers.)
    "ycb_can": dict(ycb="005_tomato_soup_can", r=0.034, h=0.102, mass=0.349, centre=(-0.0084, 0.0838)),
}
# objects the perception knows (recognition library); the mug is a distractor on the table, not grasped
YCB_LIBRARY = {
    **CUP_TYPES,
    "ycb_mug": dict(ycb="025_mug", r=0.040, h=0.081, mass=0.118, centre=(-0.0147, 0.0164)),
}


PROXY_ROBOT = """
    <!-- G1 kinematic proxy: base pose is driven by the SONIC bridge -->
    <body name="robot" mocap="true" pos="1 3 0">
      <geom name="robot_body" type="cylinder" pos="0 0 0.66" size="{ROBOT_RADIUS} 0.66" rgba="0.35 0.37 0.40 0.9" class="robot"/>
      <geom type="sphere" pos="0.02 0 1.22" size="0.11" rgba="0.15 0.15 0.15 1" class="robot"/>
      <geom type="box" pos="0.2 0 1.22" size="0.03 0.06 0.02" rgba="0.1 0.6 0.9 1" class="robot"/>
      <camera name="head" pos="0.12 0 1.25" xyaxes="0 -1 0 0.6 0 0.8" fovy="70"/>
    </body>
    <!-- right Inspire hand proxy (6 actuated DoF -> closure values) -->
    <body name="hand" mocap="true" pos="1 2.8 1.0">
      <geom name="palm" type="box" size="0.045 0.012 0.05" rgba="0.2 0.2 0.22 1" class="hand"/>
      <geom type="capsule" fromto="0.04 0.0 -0.03 0.09 0.0 -0.03" size="0.009" rgba="0.3 0.3 0.3 1" class="hand"/>
      <geom type="capsule" fromto="0.04 0.0 0.04 0.10 0.0 0.04" size="0.008" rgba="0.3 0.3 0.3 1" class="hand"/>
      <geom type="capsule" fromto="0.04 0.0 0.015 0.105 0.0 0.015" size="0.008" rgba="0.3 0.3 0.3 1" class="hand"/>
      <geom type="capsule" fromto="0.04 0.0 -0.01 0.10 0.0 -0.01" size="0.008" rgba="0.3 0.3 0.3 1" class="hand"/>
      <camera name="wrist" pos="-0.02 0.03 0.0" xyaxes="0 -1 0 0 0 1" fovy="80"/>
    </body>

"""


def _counter_with_sink(it: Item) -> list[str]:
    """Kitchen counter with a real recessed sink basin: a lower block up to the basin floor and four top slabs
    around the opening (objects dropped into the sink fall in and stay there)."""
    (cx, cy), (hx, hy) = it.pos, it.size
    x0, x1, y0, y1 = cx - hx, cx + hx, cy - hy, cy + hy
    (sx, sy), (shx, shy) = SINK_CENTER, SINK_HALF
    floor = COUNTER_TOP - SINK_DEPTH
    out = []

    def box(name, xa, xb, ya, yb, za, zb, rgba=it.rgba, cls='class="furn"'):
        out.append(f'<geom name="{name}" type="box" pos="{(xa + xb) / 2:.4f} {(ya + yb) / 2:.4f} {(za + zb) / 2:.4f}" '
                   f'size="{(xb - xa) / 2:.4f} {(yb - ya) / 2:.4f} {(zb - za) / 2:.4f}" rgba="{rgba}" {cls}/>')
    box("counter", x0, x1, y0, y1, 0.0, floor)
    box("counter_l", x0, sx - shx, y0, y1, floor, COUNTER_TOP)
    box("counter_r", sx + shx, x1, y0, y1, floor, COUNTER_TOP)
    box("counter_f", sx - shx, sx + shx, y0, sy - shy, floor, COUNTER_TOP)
    box("counter_b", sx - shx, sx + shx, sy + shy, y1, floor, COUNTER_TOP)
    # stainless basin lining (visual)
    box("sink", sx - shx, sx + shx, sy - shy, sy + shy, floor, floor + 0.002, "0.62 0.66 0.70 1",
        'contype="0" conaffinity="0" group="2"')
    return out


def build_mjcf(items: list[Item], cup_type="ycb_can", cup_rgba="0.9 0.9 0.95 1",
               with_toy=False, proxy_robot=True) -> str:
    geoms = []
    for it in items:
        z = it.h / 2
        if it.name == "counter":
            geoms += _counter_with_sink(it)
            continue
        if it.kind == "box":
            q = f'{np.cos(it.yaw/2):.5f} 0 0 {np.sin(it.yaw/2):.5f}'
            geoms.append(f'<geom name="{it.name}" type="box" pos="{it.pos[0]:.4f} {it.pos[1]:.4f} {z:.4f}" '
                         f'quat="{q}" size="{it.size[0]} {it.size[1]} {z:.4f}" rgba="{it.rgba}" class="furn"/>')
        else:
            geoms.append(f'<geom name="{it.name}" type="cylinder" pos="{it.pos[0]:.4f} {it.pos[1]:.4f} {z:.4f}" '
                         f'size="{it.size[0]} {z:.4f}" rgba="{it.rgba}" class="furn"/>')

    c = CUP_TYPES[cup_type]
    ydir = (YCB_DIR / c["ycb"] / "google_16k").as_posix()
    off = f'{-c["centre"][0]} {-c["centre"][1]} {-c["h"] / 2}'
    cup_assets = (f'<texture name="cup_tex" type="2d" file="{ydir}/texture_map.png"/>'
                  f'<material name="cup_mat" texture="cup_tex"/>'
                  f'<mesh name="cup_mesh" file="{ydir}/textured.obj" maxhullvert="64"/>')
    cx, cy = CUP_HOME
    toy = ""
    if with_toy:
        toy = ('<body name="toy" pos="2.0 3.6 0.05"><freejoint name="toy_free"/>'
               '<geom name="toy" type="box" size="0.05 0.05 0.05" rgba="0.9 0.2 0.2 1" mass="0.1" group="1"/></body>')

    proxy = PROXY_ROBOT.replace("{ROBOT_RADIUS}", str(ROBOT_RADIUS)) if proxy_robot else ""
    xml = f"""
<mujoco model="homebot_apartment">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.25 0.25 0.25" specular="0.05 0.05 0.05"/></visual>
  <default>
    <default class="furn"><geom group="0" friction="0.8 0.01 0.001"/></default>
    <default class="robot"><geom group="3" contype="0" conaffinity="0"/></default>
    <default class="hand"><geom group="2" contype="0" conaffinity="0"/></default>
  </default>
  <asset>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.90 0.87 0.80" rgb2="0.84 0.81 0.74" width="256" height="256"/>
    <material name="floor" texture="floor" texrepeat="16 12"/>
    {cup_assets}
  </asset>
  <worldbody>
    <light pos="4 3 5" dir="0 0 -1" diffuse="0.9 0.9 0.88" specular="0.2 0.2 0.2" castshadow="true"/>
    <light pos="2.2 3 2.6" dir="0 0 -1" diffuse="0.45 0.45 0.43" castshadow="false"/>
    <light pos="6.3 3 2.6" dir="0 0 -1" diffuse="0.45 0.45 0.43" castshadow="false"/>
    <light pos="6.5 4.6 2.4" dir="0 -0.3 -1" diffuse="0.35 0.35 0.33" castshadow="false"/>
    <geom name="floor" type="plane" pos="4 3 0" size="4.2 3.2 0.1" material="floor" group="0"/>
    {''.join(geoms)}
    <camera name="top" pos="4 3 9.5" xyaxes="1 0 0 0 1 0" fovy="45"/>
    <camera name="persp" pos="4 -3.2 6.5" xyaxes="1 0 0 0 0.75 0.66" fovy="55"/>

    {proxy}
    <body name="person" mocap="true" pos="0.8 2.8 0">
      <geom name="person" type="cylinder" pos="0 0 {PERSON_HEIGHT/2}" size="{PERSON_RADIUS} {PERSON_HEIGHT/2}" rgba="0.85 0.45 0.35 1" group="1" contype="0" conaffinity="0"/>
      <geom type="sphere" pos="0 0 1.62" size="0.11" rgba="0.95 0.8 0.65 1" group="1" contype="0" conaffinity="0"/>
    </body>
    <body name="pet" mocap="true" pos="-5 -5 0">
      <geom name="pet" type="box" pos="0 0 {PET_HALF[2]}" size="{PET_HALF[0]} {PET_HALF[1]} {PET_HALF[2]}" rgba="0.6 0.45 0.25 1" group="1" contype="0" conaffinity="0"/>
    </body>

    <body name="cup" pos="{cx} {cy} {TABLE_TOP + c['h']/2 + 0.002}">
      <freejoint name="cup_free"/>
      <geom name="cup" type="mesh" mesh="cup_mesh" pos="{off}" mass="{c['mass']}" group="2" rgba="1 1 1 0"
            friction="1 0.01 0.001" condim="4"/>
      <geom name="cup_visual" type="mesh" mesh="cup_mesh" pos="{off}" material="cup_mat" group="2"
            contype="0" conaffinity="0" density="0"/>
    </body>
    {toy}
  </worldbody>
</mujoco>
"""
    return xml
