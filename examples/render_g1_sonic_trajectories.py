"""Render harvested SONIC push trajectories as mp4 clips.

    MUJOCO_GL=egl python examples/render_g1_sonic_trajectories.py \
        --traj data/g1_sonic_push_test.pt --preset tour --out-dir runs/sonic_push_videos
    python examples/render_g1_sonic_trajectories.py --episodes 12 345 --speed 0.25

A REPLAY, not a re-simulation: every frame puts the recorded state of
``examples/harvest_g1_sonic_trajectories.py`` (pelvis pose, 29 joints) into a
CPU MuJoCo model of the same G1 and renders it offscreen, so a clip shows
exactly the stored trajectory. On top of it:

* the GHOST (translucent blue): the LAFAN reference SONIC is tracking, the
  clip's frame ``traj/ref_frame`` at each step, as SONIC is asked for it:
  joint angles and pelvis orientation turned by the episode's delta heading
  ``ep/dq``, at the library's snapped height. SONIC's command carries no root
  position, so after a kick the robot never walks back to the clip's path.
  The ghost is therefore drawn at the robot's own x/y: it shows the pose SONIC
  is trying to hold at each instant. Where robot and ghost overlap, SONIC
  tracks.
* the KICK: a red arrow into the pelvis along ``ep/dv_w`` (length ~ |dv|) for
  0.3 s from the onset, and the label "|dv| m/s (m |dv| N*s) from <side>"
  (random_push's sides: the side the kick comes FROM).
* the strip at the bottom: the fall margin g over the clip (g < 0 = the
  task's fall set), the onset, the cursor and the outcome (SURVIVED / FELL at
  +t s). A fallen clip holds its last recorded state (the step before the
  env's fall termination) for a second.

The camera follows the pelvis from the side of the kick: its azimuth is the
kick's direction + 90 deg, so every kick moves the robot left to right on
screen. ``--speed`` slows the clip down (0.5 = half speed: one 50 Hz step per
frame at 25 fps). ``--grid`` tiles several episodes into one clip, all aligned
at the kick.

PRESETS (``--preset``; held-out clips by default, never ``ep/ref_floor``
episodes, nor fallAndGetUp unless asked): ``tour`` = single clips across
motion categories, push sides and outcomes, plus two grids: ``frontier``
(the same |dv| near SONIC's 50 % point, survivors next to falls) and
``sweep`` (one motion type, |dv| growing). An ``index.md`` next to the clips
lists every episode rendered.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
_NV = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if os.path.exists(_NV):
  os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _NV)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)

from robot_safety_sandbox.random_push import SIDES  # noqa: E402

MOVED = {"behind": "pushed forward", "right": "pushed left",
         "front": "pushed back", "left": "pushed right"}
ROBOT_RGBA = None                       # the model's own materials
GHOST_RGBA = (0.30, 0.62, 1.00, 0.30)
KICK_RGBA = (0.92, 0.18, 0.12, 1.0)
GOOD, BAD, INK, MUTED = (40, 160, 90), (215, 50, 40), (25, 28, 33), (110, 115, 125)


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--traj", default=os.path.join(_ZOO, "data", "g1_sonic_push_test.pt"))
  p.add_argument("--out-dir", default=os.path.join(_ZOO, "runs", "sonic_push_videos"))
  p.add_argument("--preset", choices=("tour", "none"), default="tour")
  p.add_argument("--episodes", type=int, nargs="*", default=[],
                 help="also render these episode indices (single clips)")
  p.add_argument("--grid", type=int, nargs="*", default=[],
                 help="render these episodes as one grid clip")
  p.add_argument("--speed", type=float, default=0.5,
                 help="playback speed (1 = real time)")
  p.add_argument("--width", type=int, default=1280)
  p.add_argument("--height", type=int, default=720)
  p.add_argument("--no-ghost", action="store_true")
  p.add_argument("--hold-s", type=float, default=1.0,
                 help="hold the last frame this long (video time)")
  p.add_argument("--seed", type=int, default=0)
  return p.parse_args(argv)


# --- the scene -------------------------------------------------------------------

def build_model(width: int, height: int) -> mujoco.MjModel:
  """The mjlab G1 (the harvest's robot geometry) on a checker floor, lit for
  offscreen rendering."""
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
  spec = get_spec()
  spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                   builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                   rgb1=[0.88, 0.89, 0.91], rgb2=[0.76, 0.78, 0.81],
                   width=512, height=512, mark=mujoco.mjtMark.mjMARK_EDGE,
                   markrgb=[0.66, 0.68, 0.72])
  spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                   builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                   rgb1=[0.93, 0.95, 0.98], rgb2=[0.62, 0.70, 0.80],
                   width=512, height=3072)
  mat = spec.add_material(name="grid", texrepeat=[1, 1], texuniform=True,
                          reflectance=0.0)
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                          size=[40, 40, 0.05], material="grid", contype=0,
                          conaffinity=0)
  # a shadow-casting spot straight above the robot, following it (a
  # directional or grazing light's shadow map speckles the floor); the head
  # light fills in
  spec.body("pelvis").add_light(
    name="key", mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM, pos=[0.0, 0.0, 5.0],
    dir=[0.0, 0.0, -1.0], castshadow=1, cutoff=50, exponent=1,
    diffuse=[0.55, 0.55, 0.55], specular=[0.1, 0.1, 0.1])
  spec.visual.headlight.ambient = [0.42, 0.42, 0.44]
  spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
  spec.visual.headlight.specular = [0.05, 0.05, 0.05]
  spec.visual.global_.offwidth = max(width, 640)
  spec.visual.global_.offheight = max(height, 480)
  spec.visual.quality.shadowsize = 4096
  return spec.compile()


class Replay:
  """One episode's frames: robot and ghost qpos per step, the kick, labels."""

  def __init__(self, D: dict, e: int, model: mujoco.MjModel, ref: dict, lib_root_z):
    meta = D["meta"]
    self.e = e
    self.T, self.pre, self.dt = meta["T"], meta["onset_index"], meta["dt"]
    self.L = int(D["ep/length"][e])
    self.fell = bool(D["ep/fell"][e])
    self.ttf = float(D["ep/time_to_fall"][e])
    self.dv = float(D["ep/dv"][e])
    self.impulse = float(D["ep/impulse"][e])
    rel = float(D["ep/rel"][e])
    side = int(((rel + math.pi / 4) % (2 * math.pi)) // (math.pi / 2)) % 4
    self.side = SIDES[side]
    self.dv_w = D["ep/dv_w"][e].numpy().astype(np.float64)
    self.clip_name = meta["clips"][int(D["ep/clip"][e])]
    self.category = meta["categories"][int(D["ep/category"][e])]
    self.g = D["traj/g"][e, : self.L].numpy()
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    jadr = np.array([model.jnt_qposadr[names.index(n)] for n in meta["joint_names"]])
    q = np.zeros((self.L, model.nq))
    q[:, :3] = D["traj/root_pos"][e, : self.L].numpy()
    q[:, 3:7] = D["traj/root_quat"][e, : self.L].numpy()
    q[:, jadr] = D["traj/joint_pos"][e, : self.L].numpy()
    self.q = q
    self.frames = D["traj/ref_frame"][e, : self.L].numpy()
    # the ghost: the reference turned by dq, at the library's snapped height,
    # at the robot's x/y (SONIC is not asked for a root position)
    c = int(D["ep/clip"][e])
    f0 = int(D["ep/frame0"][e])
    r = ref[c]
    dq = D["ep/dq"][e].numpy().astype(np.float64)
    dz = float(lib_root_z[int(D["ep/row"][e])]) - float(r["pos"][f0, 2])
    gq = np.zeros((self.L, model.nq))
    for i, f in enumerate(self.frames):
      f = int(min(max(f, 0), r["pos"].shape[0] - 1))
      gq[i, :2] = q[i, :2]
      gq[i, 2] = r["pos"][f, 2] + dz
      qq = np.zeros(4)
      mujoco.mju_mulQuat(qq, dq, r["quat"][f])
      gq[i, 3:7] = qq
      gq[i, r["jadr"]] = r["dof"][f]
    self.ghost = gq

  def title(self) -> str:
    return (f"{self.category} · {self.clip_name} · |dv| {self.dv:.2f} m/s "
            f"from {self.side} · {'FELL' if self.fell else 'survived'}")


def load_reference(D: dict, model: mujoco.MjModel, clips: set) -> dict:
  """Root pose and joint angles of the clips at 50 fps (mjlab's MotionLoader,
  as the harvester's SonicReference builds them), CSV joint order mapped to
  this model's qpos."""
  from mjlab.scripts.csv_to_npz import MotionLoader
  from robot_safety_sandbox.envs.g1_recover.sonic import _csv_joints
  meta = D["meta"]
  names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
  jadr = np.array([model.jnt_qposadr[names.index(n)] for n in _csv_joints()])
  out = {}
  for c in sorted(clips):
    path = os.path.join(meta["csv_dir"], f"{meta['clips'][c]}.csv")
    ml = MotionLoader(path, 30, 50, "cpu")
    out[c] = {"pos": ml.motion_base_poss.numpy().astype(np.float64),
              "quat": ml.motion_base_rots.numpy().astype(np.float64),
              "dof": ml.motion_dof_poss.numpy().astype(np.float64), "jadr": jadr}
  return out


# --- drawing ---------------------------------------------------------------------

def _font(size: int, bold: bool = False):
  for path in (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
    if os.path.exists(path):
      return ImageFont.truetype(path, size)
  return ImageFont.load_default(size=size)


class Painter:
  """Renders one Replay frame: the 3-D scene plus the overlay."""

  def __init__(self, model: mujoco.MjModel, width: int, height: int, ghost: bool):
    self.m, self.w, self.h, self.ghost = model, width, height, ghost
    self.r = mujoco.Renderer(model, height, width)
    self.d = mujoco.MjData(model)
    self.d2 = mujoco.MjData(model)
    self.opt = mujoco.MjvOption()
    self.pert = mujoco.MjvPerturb()
    self.cam = mujoco.MjvCamera()
    self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    s = height / 720.0
    self.f_big, self.f_mid, self.f_small = _font(int(30 * s), True), _font(int(22 * s)), _font(int(17 * s))

  def close(self) -> None:
    self.r.close()

  def start(self, rp: Replay) -> None:
    self.look = rp.q[0, :2].copy()
    # look across the kick: it moves the robot left to right on screen
    self.cam.azimuth = math.degrees(math.atan2(rp.dv_w[1], rp.dv_w[0])) + 90.0
    self.cam.elevation = -14.0
    self.cam.distance = 4.0

  def scene(self, rp: Replay, i: int) -> np.ndarray:
    m, d, d2 = self.m, self.d, self.d2
    d.qpos[:] = rp.q[i]
    mujoco.mj_forward(m, d)
    self.look += 0.12 * (rp.q[i, :2] - self.look)
    self.cam.lookat[:] = [self.look[0], self.look[1], 0.72]
    self.r.update_scene(d, camera=self.cam, scene_option=self.opt)
    sc = self.r.scene
    if self.ghost:
      d2.qpos[:] = rp.ghost[i]
      mujoco.mj_forward(m, d2)
      n0 = sc.ngeom
      mujoco.mjv_addGeoms(m, d2, self.opt, self.pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, sc)
      for k in range(n0, sc.ngeom):
        if sc.geoms[k].category == mujoco.mjtCatBit.mjCAT_DYNAMIC:
          sc.geoms[k].rgba[:] = GHOST_RGBA
    t = (i - rp.pre) * rp.dt
    if -1e-9 <= t <= 0.3 and sc.ngeom < sc.maxgeom:
      u = rp.dv_w / max(np.linalg.norm(rp.dv_w), 1e-6)
      tip = d.qpos[:3].copy()
      tail = tip - u * (0.25 + 0.18 * rp.dv)
      g = sc.geoms[sc.ngeom]
      mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                          np.eye(3).flatten(), np.array(KICK_RGBA, np.float32))
      mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.035, tail, tip)
      sc.ngeom += 1
    return self.r.render()

  def overlay(self, img: np.ndarray, rp: Replay, i: int, compact: bool = False) -> np.ndarray:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    s = h / 720.0
    t = (i - rp.pre) * rp.dt
    pad = int(18 * s)
    # header
    if not compact:
      dr.rounded_rectangle([pad - 8, pad - 6, int(w * 0.62), pad + int(92 * s)], radius=10,
                           fill=(255, 255, 255, 200))
      dr.text((pad, pad), f"SONIC tracking LAFAN · {rp.category} · {rp.clip_name}",
              font=self.f_mid, fill=INK)
      dr.text((pad, pad + int(32 * s)),
              f"kick |dv| = {rp.dv:.2f} m/s  (≈ {rp.impulse:.0f} N·s)  from {rp.side}"
              f"  ({MOVED[rp.side]})", font=self.f_mid, fill=BAD)
      dr.text((pad, pad + int(64 * s)),
              "grey: robot (SONIC in control)   blue ghost: LAFAN reference",
              font=self.f_small, fill=MUTED)
    else:
      dr.rounded_rectangle([6, 4, w - 6, int(64 * s)], radius=8, fill=(255, 255, 255, 200))
      dr.text((12, 8), f"{rp.category} · {rp.dv:.2f} m/s from {rp.side}",
              font=self.f_mid, fill=INK)
      out = "FELL" if rp.fell else "survived"
      dr.text((12, 8 + int(28 * s)), out, font=self.f_mid, fill=BAD if rp.fell else GOOD)
    # clock
    clock = f"t = {t:+.2f} s"
    tw = dr.textlength(clock, font=self.f_big)
    dr.text((w - tw - pad, pad), clock, font=self.f_big, fill=INK)
    if -1e-9 <= t <= 0.3:
      kt = "KICK"
      kw = dr.textlength(kt, font=self.f_big)
      dr.text((w - kw - pad, pad + int(40 * s)), kt, font=self.f_big, fill=BAD)
    # the g strip
    if not compact:
      self._strip(dr, rp, i, w, h, s)
    if i >= rp.L - 1 and not compact:
      # under the clock, clear of the robot
      msg = (f"FELL at +{rp.ttf:.2f} s" if rp.fell else "SURVIVED")
      mw = dr.textlength(msg, font=self.f_big)
      col = BAD if rp.fell else GOOD
      y = pad + int(84 * s)
      dr.rounded_rectangle([w - mw - pad - 12, y - 6, w - pad + 10, y + int(42 * s)],
                           radius=10, fill=(255, 255, 255, 220))
      dr.text((w - mw - pad, y), msg, font=self.f_big, fill=col)
    return np.asarray(im)

  def _strip(self, dr, rp: Replay, i: int, w: int, h: int, s: float) -> None:
    x0, x1 = int(70 * s), w - int(30 * s)
    y0, y1 = h - int(96 * s), h - int(22 * s)
    dr.rounded_rectangle([x0 - 60 * s, y0 - 12 * s, x1 + 14 * s, y1 + 10 * s], radius=10,
                         fill=(255, 255, 255, 200))
    T = rp.T

    def X(j):
      return x0 + (x1 - x0) * j / max(T - 1, 1)

    gmin, gmax = -1.0, 1.6

    def Y(g):
      g = min(max(g, gmin), gmax)
      return y1 - (y1 - y0) * (g - gmin) / (gmax - gmin)

    dr.line([X(0), Y(0.0), X(T - 1), Y(0.0)], fill=(*BAD, 160), width=1)
    dr.text((x0 - 52 * s, Y(0.0) - 9 * s), "g = 0", font=self.f_small, fill=BAD)
    dr.text((x0 - 52 * s, y0 - 6 * s), "fall", font=self.f_small, fill=MUTED)
    dr.text((x0 - 52 * s, y0 + 10 * s), "margin", font=self.f_small, fill=MUTED)
    pts = [(X(j), Y(float(rp.g[j]))) for j in range(min(i + 1, rp.L)) if np.isfinite(rp.g[j])]
    if len(pts) > 1:
      dr.line(pts, fill=(*INK, 255), width=max(2, int(2 * s)))
    xp = X(rp.pre)
    dr.line([xp, y0, xp, y1], fill=(*BAD, 220), width=max(2, int(2 * s)))
    dr.text((xp + 4, y0 - 4 * s), "kick", font=self.f_small, fill=BAD)
    if rp.fell:
      xf = X(rp.L)
      dr.line([xf, y0, xf, y1], fill=(*BAD, 120), width=1)
    xc = X(min(i, T - 1))
    dr.line([xc, y0, xc, y1], fill=(*INK, 120), width=1)


def render_single(path: str, rp: Replay, painter: Painter, fps: float, hold_s: float) -> None:
  import imageio.v2 as imageio
  painter.start(rp)
  with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                          macro_block_size=8, ffmpeg_log_level="error") as wr:
    frame = None
    for i in range(rp.L):
      frame = painter.overlay(painter.scene(rp, i), rp, i)
      wr.append_data(frame)
    for _ in range(int(round(hold_s * fps))):
      wr.append_data(frame)


def render_grid(path: str, rps: list, painter: Painter, fps: float, hold_s: float,
                cols: int, title: str) -> None:
  """Tiles, aligned at the kick (every trajectory has it at index pre)."""
  import imageio.v2 as imageio
  T = rps[0].T
  rows = math.ceil(len(rps) / cols)
  tw, th = painter.w, painter.h
  band = int(46 * th / 360)
  W, H = tw * cols, th * rows + band
  H += (-H) % 8
  for rp in rps:
    rp._cam = None
  f_title = _font(int(26 * th / 360), True)
  with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                          macro_block_size=8, ffmpeg_log_level="error") as wr:
    last = None
    for i in range(T):
      canvas = Image.new("RGB", (W, H), (245, 246, 248))
      for k, rp in enumerate(rps):
        j = min(i, rp.L - 1)
        if rp._cam is None:
          painter.start(rp)
          rp._cam = (painter.look.copy(), painter.cam.azimuth)
        painter.look, painter.cam.azimuth = rp._cam
        img = painter.scene(rp, j)
        rp._cam = (painter.look.copy(), painter.cam.azimuth)
        img = painter.overlay(img, rp, j, compact=True)
        tile = Image.fromarray(img)
        if rp.fell and i >= rp.L:
          dr = ImageDraw.Draw(tile, "RGBA")
          dr.rectangle([0, 0, tw, th], fill=(215, 50, 40, 40))
        canvas.paste(tile, ((k % cols) * tw, band + (k // cols) * th))
      dr = ImageDraw.Draw(canvas)
      t = (i - rps[0].pre) * rps[0].dt
      dr.text((12, 8), f"{title}    t = {t:+.2f} s from the kick", font=f_title, fill=INK)
      last = np.asarray(canvas)
      wr.append_data(last)
    for _ in range(int(round(hold_s * fps))):
      wr.append_data(last)


# --- choosing episodes -------------------------------------------------------------

class Picker:
  def __init__(self, D: dict, seed: int):
    self.D, self.g = D, np.random.default_rng(seed)
    cats = D["meta"]["categories"]
    self.cats = cats
    self.cat = D["ep/category"].numpy()
    self.dv = D["ep/dv"].numpy()
    self.fell = D["ep/fell"].numpy()
    rel = D["ep/rel"].numpy()
    self.side = (((rel + math.pi / 4) % (2 * math.pi)) // (math.pi / 2)).astype(int) % 4
    jerr = D["ep/pre_jerr"].numpy()
    # representative: no reference floor contact, tracking at least median well
    self.ok = ~D["ep/ref_floor"].numpy() & (jerr <= np.median(jerr))
    self.used: set = set()

  def pick(self, cat=None, dv=(0, 9), side=None, fell=None, allow_fgu=False) -> int | None:
    m = self.ok & (self.dv >= dv[0]) & (self.dv < dv[1])
    if cat is not None:
      m &= self.cat == self.cats.index(cat)
    elif not allow_fgu and "fallAndGetUp" in self.cats:
      m &= self.cat != self.cats.index("fallAndGetUp")
    if side is not None:
      m &= self.side == SIDES.index(side)
    if fell is not None:
      m &= self.fell == fell
    idx = [int(i) for i in np.nonzero(m)[0] if int(i) not in self.used]
    if not idx:
      return None
    e = int(self.g.choice(idx))
    self.used.add(e)
    return e


def tour(pk: Picker):
  """(name, episode) singles and (name, title, cols, episodes) grids."""
  singles = [
    ("walk_small_kick", dict(cat="walk", dv=(0.8, 1.2), fell=False)),
    ("walk_frontier_from_behind_survived", dict(cat="walk", dv=(2.5, 3.0), side="behind", fell=False)),
    ("walk_frontier_from_behind_fell", dict(cat="walk", dv=(2.5, 3.0), side="behind", fell=True)),
    ("walk_frontier_from_right_survived", dict(cat="walk", dv=(2.5, 3.0), side="right", fell=False)),
    ("walk_frontier_from_front_fell", dict(cat="walk", dv=(2.5, 3.0), side="front", fell=True)),
    ("run_frontier_survived", dict(cat="run", dv=(2.3, 3.0), fell=False)),
    ("run_frontier_fell", dict(cat="run", dv=(2.3, 3.0), fell=True)),
    ("dance_frontier_survived", dict(cat="dance", dv=(2.3, 3.0), fell=False)),
    ("jumps_frontier_fell", dict(cat="jumps", dv=(2.3, 3.0), fell=True)),
    ("fight_big_kick_survived", dict(cat="fight", dv=(3.3, 4.2), fell=False)),
    ("sprint_frontier_fell", dict(cat="sprint", dv=(2.3, 3.0), fell=True)),
    ("walk_huge_kick_fell", dict(cat="walk", dv=(4.4, 5.0), fell=True)),
  ]
  out_s = []
  for name, crit in singles:
    e = pk.pick(**crit)
    if e is not None:
      out_s.append((name, e))
  frontier = [pk.pick(dv=(2.6, 3.0), fell=f) for f in (False, False, False, True, True, True)]
  sweep = [pk.pick(cat="walk", dv=(v, v + 0.25), side="behind") for v in (0.5, 1.5, 2.25, 2.75, 3.25, 4.0)]
  grids = [("grid_frontier_2.6-3.0ms", "same kick (2.6-3.0 m/s), top: survived, bottom: fell", 3,
            [e for e in frontier if e is not None]),
           ("grid_sweep_walk_from_behind", "walking, kicked from behind: 0.5 -> 4.25 m/s", 3,
            [e for e in sweep if e is not None])]
  return out_s, grids


def main(argv=None):
  args = _parse(argv)
  os.makedirs(args.out_dir, exist_ok=True)
  D = torch.load(args.traj, map_location="cpu", weights_only=False)
  meta = D["meta"]
  pk = Picker(D, args.seed)
  singles = [(f"ep{e}", e) for e in args.episodes]
  grids = [("grid_" + "_".join(map(str, args.grid)), "selected episodes", 3, list(args.grid))] if args.grid else []
  if args.preset == "tour":
    s, g = tour(pk)
    singles += s
    grids += g
  need = {e for _, e in singles} | {e for *_, es in grids for e in es}
  if not need:
    raise SystemExit("nothing to render (--preset none and no --episodes / --grid)")
  from robot_safety_sandbox.envs.g1_recover.states import load_bank
  lib = load_bank(meta["library"], "cpu")
  model = build_model(args.width, args.height)
  ref = load_reference(D, model, {int(D["ep/clip"][e]) for e in need})
  rps = {e: Replay(D, e, model, ref, lib["root_z"]) for e in sorted(need)}
  fps = 50.0 * args.speed
  lines = [f"# SONIC push clips\n\nFrom `{args.traj}` ({meta.get('created', '')}); "
           f"playback {args.speed:g}x ({fps:g} fps of 50 Hz steps).\n"]
  painter = Painter(model, args.width, args.height, not args.no_ghost)
  for k, (name, e) in enumerate(singles):
    path = os.path.join(args.out_dir, f"{k + 1:02d}_{name}.mp4")
    render_single(path, rps[e], painter, fps, args.hold_s)
    lines.append(f"- `{os.path.basename(path)}`: episode {e}: {rps[e].title()}")
    print(f"[render] {path}: {rps[e].title()}", flush=True)
  if grids:
    gp = Painter(model, args.width // 2, args.height // 2, not args.no_ghost)
    for name, title, cols, es in grids:
      if not es:
        continue
      path = os.path.join(args.out_dir, f"{name}.mp4")
      render_grid(path, [rps[e] for e in es], gp, fps, args.hold_s, cols, title)
      lines.append(f"- `{os.path.basename(path)}`: {title}: episodes "
                   + ", ".join(f"{e} ({rps[e].title()})" for e in es))
      print(f"[render] {path}: {title}", flush=True)
    gp.close()
  painter.close()
  with open(os.path.join(args.out_dir, "index.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
  print(f"[render] {len(singles)} clips + {len(grids)} grids -> {args.out_dir}")


if __name__ == "__main__":
  main()
