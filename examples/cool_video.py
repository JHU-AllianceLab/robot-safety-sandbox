"""Video of a cooling policy taking over with hot ankles: a grid of robots, each
with its ankle temperatures (the user, 2026-11-19: "render another video showing
how the robot would move under the cooling policy"), optionally after a
fallback policy has run first ("run fallback policy for awhile and transit to
this cooling policy").

Every env is one env of ``--task`` (default ``g1_cool_thermal70``, the pilot's
task) on a SONIC handover row drawn by the task's own spawn. Without
``--fallback`` the cooling policy (``--policy``, deterministic) acts from
t = 0; with it, the ``--fallback`` policy acts until ``--switch-s`` and the
cooling policy from then on (every env switches at the same step; the cooling
policy's first observation carries the fallback's last action, as a handover
row carries SONIC's). A ``--switch-s`` at or past ``--duration-s`` runs the
fallback throughout (a reference).

The start temperatures are ``cool_curve.py``'s: every ankle winding (pitch and
roll, both legs) at ``--ankle-c``, the ankle housings at ``--housing-c``, the
structures at ``--structure-c``, the other motors at ``--rest-c``. The task's
random minor pushes stay on (``--no-push`` turns them off); the handover row's
own pending kick still finishes. An env that falls or overheats stops there
(the env's auto-reset would redraw it): its panel freezes on its last frame
under the outcome.

Video (the first ``--panels`` envs): each panel is the robot from a 3/4 front
view fixed at its spawn heading (tracking the pelvis), the time, who acts, the
hottest ankle winding, ``PUSH`` while a push acts, and under it a strip with
the hottest ankle winding so far (green while the fallback acts, blue under
the cooling policy), the 70 C limit (red dashes), the switch (white) and the
push onsets (orange ticks). One mp4 at real time (50 fps), the last frame held.

Stats (every env, ``--num-envs``; ``--no-video`` for stats only): who fell or
overheated when (before the switch, within 2 s after it, later), the RMS
per-joint action change at the switch against each policy's own step-to-step
change around it, and the median hottest ankle winding of the envs still
standing over time. Written to ``<out>_summary.txt`` and ``<out>_stats.pt``.

    MUJOCO_GL=egl python examples/cool_video.py \\
      --policy runs/g1_cool/pilot_s0/g1_cool_thermal70/final_model.zip \\
      --fallback runs/shac_thermal70/shac_s0/g1_recover_thermal70_safetyshac1p/final_model.zip \\
      --switch-s 10 --out runs/g1_cool/video/switch10.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MASS_KG = 33.0                       # build_cool_bank.py's G1 mass, for push dv
BG, INK, INK2 = (22, 22, 26), (240, 240, 240), (170, 170, 175)
BLUE, RED, ORANGE, GREEN = (95, 165, 255), (230, 70, 70), (245, 150, 40), (70, 200, 110)
FB_SHADE = (28, 44, 34)
STRIP_H = 84
T_LO, T_HI = 30.0, 72.0              # the strip's temperature axis, C
AFTER_S = 2.0                        # "fell right after the switch" window, s


def temp_color(c: float):
  """Blue at <= 40 C to red at >= 65 C (the readout only; the strip is one hue)."""
  u = min(max((c - 40.0) / 25.0, 0.0), 1.0)
  return tuple(int(round(a + (b - a) * u)) for a, b in zip(BLUE, RED))


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--policy", required=True, help="the cooling policy zip (PPO or SHAC twin)")
  p.add_argument("--fallback", default=None,
                 help="a policy zip acting before --switch-s (the cooling policy after)")
  p.add_argument("--switch-s", type=float, default=10.0)
  p.add_argument("--label", default=None)
  p.add_argument("--task", default="g1_cool_thermal70")
  p.add_argument("--panels", type=int, default=4)
  p.add_argument("--num-envs", type=int, default=0,
                 help="envs simulated (stats over all; the first --panels are rendered); "
                      "0: --panels")
  p.add_argument("--no-video", action="store_true", help="stats only: no render, no mp4")
  p.add_argument("--cols", type=int, default=2)
  p.add_argument("--duration-s", type=float, default=30.0)
  p.add_argument("--ankle-c", type=float, default=65.0)
  p.add_argument("--housing-c", type=float, default=35.0)
  p.add_argument("--structure-c", type=float, default=30.0)
  p.add_argument("--rest-c", type=float, default=30.0)
  p.add_argument("--no-push", action="store_true")
  p.add_argument("--size", type=int, default=360, help="panel width = height, px")
  p.add_argument("--distance", type=float, default=2.6)
  p.add_argument("--elevation", type=float, default=-12.0)
  p.add_argument("--view-deg", type=float, default=35.0,
                 help="camera azimuth off the robot's front, deg (0: head-on)")
  p.add_argument("--hold-s", type=float, default=2.0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", required=True,
                 help="mp4 path (or, with --no-video, the prefix of the summary files)")
  a = p.parse_args()
  if a.size % 2:
    p.error("--size must be even (libx264)")
  video = not a.no_video
  n_vid = int(a.panels) if video else 0
  n = max(int(a.num_envs), n_vid, 1)
  prefix = a.out[:-4] if a.out.endswith(".mp4") else a.out

  from PIL import Image, ImageDraw

  import filter_video as fv
  from cool_curve import _last_action_slice, load_policy
  from robot_safety_sandbox.base import MjlabTensorSafetyEnv
  from robot_safety_sandbox.envs.g1_recover.spawn import RecoverSpawn
  from robot_safety_sandbox.envs.g1_recover.thermal import thermal_state
  from robot_safety_sandbox.registry import CUMULATIVE, spec

  dev = a.device
  s = spec(a.task)

  def builder(play: bool = False, **kw):
    cfg = s.cfg_builder(play=play, **kw)
    fv.video_cfg(cfg, a.size, a.size)
    cfg.viewer.distance, cfg.viewer.elevation = float(a.distance), float(a.elevation)
    return cfg

  kw = dict(s.kwargs)
  kw.setdefault("dense_reward", s.mode == CUMULATIVE)
  env = MjlabTensorSafetyEnv(
    n, dev, cfg_builder=builder, margin_fn=s.margin_fn, ctrl_dim=s.ctrl_dim,
    dstb_dim=s.dstb_dim, adversary=False, end_criterion=s.end_criterion,
    cfg_overrides={"episode_length_s": a.duration_s + 5.0},
    render_mode="rgb_array" if video else None, **kw)
  mj = env.mj
  push = env._push
  pushes_on = push is not None and not a.no_push
  if push is not None and a.no_push:
    push.max_pushes = 0
  rend = getattr(mj, "_offline_renderer", None)
  dt = float(mj.step_dt)
  steps = int(round(a.duration_s / dt))
  fps = int(round(1.0 / dt))
  bound = float(env.action_bound)
  la = _last_action_slice(mj, env.obs_key)
  st = thermal_state(mj)
  names = list(st.names)
  ank = [i for i, nm in enumerate(names) if "ankle" in nm]
  pols = {"cooling": load_policy(a.policy, dev)}
  k_sw = 0
  if a.fallback:
    pols["fallback"] = load_policy(a.fallback, dev)
    k_sw = int(round(a.switch_s / dt))
  switches = 0 < k_sw < steps

  def who(k: int) -> str:
    return "fallback" if k < k_sw else "cooling"

  if a.label is None:
    if not a.fallback:
      a.label = "Cooling policy (first run, 500M steps) from the handover"
    elif switches:
      a.label = f"Fallback (SHAC) for {a.switch_s:g} s, then the cooling policy (first run)"
    else:
      a.label = "Fallback (SHAC) throughout"

  torch.manual_seed(a.seed)
  obs = env.reset()
  Tw = torch.full((n, len(names)), a.rest_c, device=dev)
  Th, Ts = Tw.clone(), Tw.clone()
  Tw[:, ank] = a.ankle_c
  Th[:, ank] = min(a.housing_c, a.ankle_c)
  Ts[:, ank] = min(a.structure_c, a.housing_c, a.ankle_c)
  st.set(None, Tw, Th, Ts)

  # the spawn rows (labels) and each robot's heading at the spawn (camera azimuth)
  labels = [f"Robot {i + 1}" for i in range(n)]
  try:
    spawn = next(c.func for cfgs in mj.event_manager._mode_term_cfgs.values() for c in cfgs
                 if isinstance(getattr(c, "func", None), RecoverSpawn))
    rows = mj._zoo_spawn_row.cpu()
    bank = spawn.bank
    cats = list(bank["meta"]["categories"])
    for i in range(n_vid):
      r = int(rows[i])
      cat = cats[int(bank["category"][r])]
      imp = float(torch.nan_to_num(bank["push_impulse"][r].float(), 0.0))
      labels[i] += f": {cat}, kick {imp / MASS_KG:.1f} m/s"
  except Exception as e:                      # labels are cosmetic
    print(f"[video] no spawn labels ({type(e).__name__}: {e})", flush=True)
  q = mj.scene["robot"].data.root_link_quat_w.cpu()          # (w, x, y, z)
  yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                    1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  azim = [math.degrees(float(y)) + 180.0 + a.view_deg for y in yaw]

  hist = np.full((n, steps + 1), np.nan)                     # hottest ankle winding, C
  dact = np.full((n, steps), np.nan)                         # RMS per-joint action change
  onsets: list[list[int]] = [[] for _ in range(n_vid)]
  alive = np.ones(n, dtype=bool)
  end_k = np.full(n, -1)
  reason = np.array([""] * n, dtype=object)
  last_img: list = [None] * n_vid
  pushing = np.zeros(n, dtype=bool)
  push_dv = np.zeros(n)

  # --- drawing ------------------------------------------------------------------------
  big, mid, small = fv._font(24), fv._font(18), fv._font(14)
  W = a.size
  cols = max(1, min(a.cols, max(n_vid, 1)))
  nrows = int(math.ceil(max(n_vid, 1) / cols))
  push_txt = ("no pushes" if not pushes_on else
              f"minor pushes (~{float(env.force_max) * push.duration * dt / MASS_KG:.1f} m/s) "
              f"every {push._interval[0] * dt:.0f}-{push._interval[1] * dt:.0f} s, as in "
              f"training")
  banner_lines = [
    a.label,
    f"{n_vid} random SONIC handovers at t = 0; all 4 ankle windings start at "
    f"{a.ankle_c:.0f} C (housing {a.housing_c:.0f} C); {push_txt}",
    "strip: hottest ankle winding" + (" (green: fallback acting, blue: cooling policy)"
                                      if a.fallback else " (blue)")
    + ", 70 C limit (red dashes), pushes (orange)"]

  def wrap(text: str, font, width: int) -> list[str]:
    d0 = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    out, cur = [], ""
    for w_ in text.split(" "):
      nxt = (cur + " " + w_).strip()
      if d0.textlength(nxt, font=font) > width and cur:
        out.append(cur)
        cur = w_
      else:
        cur = nxt
    return out + [cur]

  lines = [(banner_lines[0], mid, INK)] + [(ln, small, INK2) for b in banner_lines[1:]
                                           for ln in wrap(b, small, W * cols - 20)]
  bh = 12 + 22 * len(lines)
  bh += bh % 2
  banner = Image.new("RGB", (W * cols, bh), BG)
  d = ImageDraw.Draw(banner)
  for j, (line, font, col) in enumerate(lines):
    d.text((10, 6 + 22 * j), line, fill=col, font=font)
  banner = np.asarray(banner)

  def render(i: int) -> np.ndarray:
    rend._env_ids = (int(i),)
    rend._cam.azimuth = azim[i]
    return np.asarray(env.render())

  def panel(i: int, k: int) -> np.ndarray:
    img = Image.fromarray(last_img[i])
    done = end_k[i] >= 0
    if done:
      img = Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), 0.45)
    dr = ImageDraw.Draw(img)
    dr.text((8, 6), labels[i], fill=INK, font=small, stroke_width=2, stroke_fill=BG)
    tk = k if not done else int(end_k[i])
    dr.text((8, 26), f"t = {tk * dt:4.1f} s", fill=INK, font=mid, stroke_width=2,
            stroke_fill=BG)
    if a.fallback:
      ph = who(min(tk, steps - 1))
      dr.text((8, 48), "FALLBACK" if ph == "fallback" else "COOLING", font=mid,
              fill=GREEN if ph == "fallback" else BLUE, stroke_width=2, stroke_fill=BG)
    seen = np.flatnonzero(~np.isnan(hist[i, :tk + 1]))
    c = hist[i, seen[-1]] if seen.size else np.nan
    if not np.isnan(c):
      txt = f"ankle {c:.1f} C"
      w_ = dr.textlength(txt, font=big)
      dr.text((W - 8 - w_, 22), txt, fill=temp_color(c), font=big, stroke_width=2,
              stroke_fill=BG)
    if switches and not done and 0 <= tk - k_sw < fps:
      txt = "SWITCH TO COOLING"
      w_ = dr.textlength(txt, font=big)
      dr.text(((W - w_) / 2, 76), txt, fill=BLUE, font=big, stroke_width=2, stroke_fill=BG)
    if not done and pushing[i]:
      txt = f"PUSH {push_dv[i]:.1f} m/s"
      w_ = dr.textlength(txt, font=big)
      dr.text(((W - w_) / 2, W - 40), txt, fill=ORANGE, font=big, stroke_width=2,
              stroke_fill=BG)
    if done:
      txt = f"{reason[i].upper()} at {end_k[i] * dt:.1f} s"
      w_ = dr.textlength(txt, font=big)
      dr.text(((W - w_) / 2, W / 2 - 14), txt, fill=RED, font=big, stroke_width=2,
              stroke_fill=BG)
    # the strip
    sp = Image.new("RGB", (W, STRIP_H), BG)
    ds = ImageDraw.Draw(sp)
    x0, x1, y0, y1 = 34, W - 8, 6, STRIP_H - 16

    def xy(kk, cc):
      return (x0 + (x1 - x0) * kk / steps,
              y1 - (y1 - y0) * (min(max(cc, T_LO), T_HI) - T_LO) / (T_HI - T_LO))

    if a.fallback:
      xs = xy(min(k_sw, steps), T_LO)[0]
      ds.rectangle([x0, y0, xs, y1], fill=FB_SHADE)
      if switches:
        ds.line([(xs, y0), (xs, y1)], fill=INK2, width=1)
    for cc in (40.0, 50.0, 60.0):
      yy = xy(0, cc)[1]
      ds.line([(x0, yy), (x1, yy)], fill=(50, 50, 58))
      ds.text((4, yy - 8), f"{cc:.0f}", fill=INK2, font=small)
    yy = xy(0, 70.0)[1]
    for xx in range(x0, x1, 10):
      ds.line([(xx, yy), (min(xx + 5, x1), yy)], fill=RED, width=1)
    for kk in onsets[i]:
      if kk <= tk:
        xx = xy(kk, T_LO)[0]
        ds.line([(xx, y1 - 8), (xx, y1 + 2)], fill=ORANGE, width=2)
    h = hist[i, :tk + 1]
    for lo, hi, col in ((0, min(k_sw, tk), GREEN), (min(k_sw, tk), tk, BLUE)):
      pts = [xy(kk, h[kk]) for kk in list(range(lo, hi + 1, 2)) + [hi]
             if kk <= tk and not np.isnan(h[kk])]
      if len(pts) > 1:
        ds.line(pts, fill=col, width=2)
    ds.text((x0, STRIP_H - 15), "0 s", fill=INK2, font=small)
    lab = f"{a.duration_s:.0f} s"
    ds.text((x1 - ds.textlength(lab, font=small), STRIP_H - 15), lab, fill=INK2, font=small)
    return np.concatenate([np.asarray(img), np.asarray(sp)], axis=0)

  def frame(k: int) -> np.ndarray:
    tiles = [panel(i, k) for i in range(n_vid)]
    blank = np.full_like(tiles[0], BG[0])
    tiles += [blank] * (nrows * cols - n_vid)
    grid = np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1)
                           for r in range(nrows)], axis=0)
    return np.concatenate([banner, grid], axis=0)

  def record(k: int) -> None:
    hot = st.Tw[:, ank].amax(1).cpu().numpy()
    hist[alive, k] = hot[alive]
    for i in range(n_vid):
      if alive[i]:
        last_img[i] = render(i)

  # --- the run ------------------------------------------------------------------------
  writer = None
  if video:
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    writer = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=7,
                                macro_block_size=1)
  os.makedirs(os.path.dirname(os.path.abspath(prefix)) or ".", exist_ok=True)
  tm = mj.termination_manager
  prev = None
  try:
    record(0)
    if writer is not None:
      writer.append_data(frame(0))
    for k in range(steps):
      model, norm, twin = pols[who(k)]
      o = obs
      if twin:
        o = obs.clone()
        o[:, la] = o[:, la].clamp(-bound, bound)
      with torch.no_grad():
        act = torch.clamp(model.policy._predict(norm(o), deterministic=True), -bound, bound)
      if prev is not None:
        dc = (act - prev).pow(2).mean(1).sqrt().cpu().numpy()
        dact[alive, k] = dc[alive]
      prev = act
      obs, _r, dones, _to, _l = env.step_tensor(act)
      dn = dones.bool().cpu().numpy()
      if push is not None:
        f = push.cur_force_w.norm(dim=1).cpu().numpy()
        on = f > 1e-6
        for i in np.flatnonzero(on[:n_vid] & ~pushing[:n_vid] & alive[:n_vid]):
          onsets[i].append(k + 1)
        pushing = on
        push_dv = f * push.duration * dt / MASS_KG
      new = np.flatnonzero(dn & alive)
      if new.size:
        terms = {t: tm.get_term(t).bool().cpu().numpy() for t in tm.active_terms}
        for i in new:
          why = [t for t, v in terms.items() if v[i]]
          reason[i] = ("overheated" if any("heat" in t for t in why) else
                       "timed out" if why and all(t == "time_out" for t in why) else "fell")
          end_k[i] = k + 1
          alive[i] = False
          if i < n_vid:
            print(f"[video] {labels[i]}: {reason[i]} at {(k + 1) * dt:.2f} s (terms {why})",
                  flush=True)
      record(k + 1)
      if writer is not None:
        writer.append_data(frame(k + 1))
      if (k + 1) % (10 * fps) == 0:
        print(f"[video] t = {(k + 1) * dt:.0f} s, standing {int(alive.sum())}/{n}", flush=True)
    if writer is not None:
      last = frame(steps)
      for _ in range(int(round(a.hold_s * fps))):
        writer.append_data(last)
  finally:
    if writer is not None:
      writer.close()

  # --- the summary --------------------------------------------------------------------
  out = [f"{a.label}", f"cooling policy {a.policy}"]
  if a.fallback:
    out.append(f"fallback {a.fallback}, switch at {a.switch_s:g} s"
               + ("" if switches else " (never: fallback throughout)"))
  out.append(f"task {a.task}, seed {a.seed}, {n} envs, {a.duration_s:g} s, ankles "
             f"{a.ankle_c:g} C, housing {a.housing_c:g} C, pushes {'on' if pushes_on else 'off'}")
  ended = end_k >= 0
  t_end = end_k * dt
  ref_k = k_sw if switches else 0
  ref_s = ref_k * dt
  tag = "switch" if switches else ("start" if ref_k == 0 else "")
  fell = ended & (reason == "fell")
  hot = ended & (reason == "overheated")
  out.append(f"standing at {a.duration_s:g} s: {int((~ended).sum())}/{n} "
             f"({(~ended).mean():.1%}); fell {int(fell.sum())}, overheated {int(hot.sum())}")
  if switches:
    pre = ended & (end_k <= k_sw)
    win = ended & (end_k > k_sw) & (t_end <= ref_s + AFTER_S)
    late = ended & (t_end > ref_s + AFTER_S)
    n_sw = int(n - pre.sum())
    out.append(f"  ended before the switch (fallback acting): {int(pre.sum())}/{n} "
               f"({pre.mean():.1%})")
    out.append(f"  of the {n_sw} standing at the switch: ended within {AFTER_S:g} s after it "
               f"{int(win.sum())} ({win.sum() / max(n_sw, 1):.1%}), later "
               f"{int(late.sum())} ({late.sum() / max(n_sw, 1):.1%})")
  else:
    early = ended & (t_end <= AFTER_S)
    out.append(f"  ended within {AFTER_S:g} s of the start: {int(early.sum())}/{n} "
               f"({early.mean():.1%}), later {int((ended & ~early).sum())}")
  if switches:
    jump = dact[:, k_sw]
    jump = jump[~np.isnan(jump)]
    before = np.nanmedian(dact[:, max(1, k_sw - fps):k_sw], axis=1)
    after = np.nanmedian(dact[:, k_sw + 1:k_sw + 1 + fps], axis=1)
    out.append(f"  action change at the switch, RMS per joint: median {np.median(jump):.2f}, "
               f"p90 {np.percentile(jump, 90):.2f}, max {jump.max():.2f} (the 1 s before, "
               f"fallback step to step: median {np.nanmedian(before):.3f}; the 1 s after, "
               f"cooling: {np.nanmedian(after):.3f}; SONIC -> cooling at a handover was ~2.8)")
  marks = sorted({0.0, 5.0, 10.0, 20.0, a.duration_s}
                 | ({ref_s, ref_s + 1.0, ref_s + 2.0, ref_s + 5.0, ref_s + 10.0}
                    if switches else set()))
  out.append("  hottest ankle winding, median (p10-p90) of the envs standing then:")
  for t_ in marks:
    kk = int(round(t_ / dt))
    if kk > steps:
      continue
    h = hist[:, kk]
    h = h[~np.isnan(h)]
    if h.size:
      out.append(f"    t {t_:5.1f} s ({who(min(kk, steps - 1))}): {np.median(h):5.1f} "
                 f"({np.percentile(h, 10):.1f}-{np.percentile(h, 90):.1f}) C, {h.size} standing")
  for i in range(n_vid):
    o_ = "standing" if end_k[i] < 0 else f"{reason[i]} at {end_k[i] * dt:.2f} s"
    out.append(f"  {labels[i]}: {o_}; pushes {len(onsets[i])}")
  txt = "\n".join(out)
  with open(prefix + "_summary.txt", "w") as fh:
    fh.write(txt + "\n")
  torch.save({"args": vars(a), "dt": dt, "k_switch": k_sw, "hist": hist, "dact": dact,
              "end_k": end_k, "reason": list(reason)}, prefix + "_stats.pt")
  print(txt, flush=True)
  print(f"[video] wrote {prefix}_summary.txt" + (f", {a.out}" if video else ""), flush=True)


if __name__ == "__main__":
  main()
