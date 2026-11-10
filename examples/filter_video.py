"""Videos of chosen push-suite episodes under several filters, side by side, with who acts and the margins.

Used by ``eval_sonic_qcbf.py --video-episodes N ...``: the episodes render inside
the evaluation's own batch, in their own env slots (the renderer is pointed at
the slot of each episode in turn), so every panel is that run's episode, with
its startup domain randomization, and its outcome is the one ``push.json``
counts.

Per episode, one mp4: a banner (clip, kick) over one panel per condition
(``--video-conditions``, in order). A panel is the robot from the spawn until
it survives the window or falls (the last frame held under the outcome), its
label, the time since the kick, who is acting now (SONIC or the filter's pi),
and under it a strip that fills in as the episode runs:

  bar     per control step, who acted: blue SONIC, green pi (SHAC)
  white   the real margin g after the step (compose(g_fall), < 0: failure set)
  orange  the filter's verdict at that step (wmmin: predicted worst margin over
          H - eps_H; wmrollout / rollout: predicted worst margin over H). Below
          0 the filter hands the step to pi. A latched filter shows its raw
          verdict; its bar stays green after the first hand-over.

Writes ``<video dir>/ep<NNN>_<category>_<dv>ms_<direction>_<outcomes>.mp4`` (one
letter per panel, S survived / F fell), ``reel.mp4`` (all, end to end) and
``summary.json`` / ``summary.txt`` (per episode and panel: outcome, first
hand-over, share of post-kick steps under pi, hand-overs either way).
"""

from __future__ import annotations

import io
import json
import math
import os

import numpy as np

DIRS_8 = ("forward", "forward-left", "left", "back-left", "back", "back-right", "right",
          "forward-right")
BLUE, GREEN, RED, ORANGE = (60, 110, 225), (40, 175, 85), (215, 55, 55), (240, 145, 30)
WHITE, GREY = (240, 240, 240), (120, 120, 125)
BANNER_H, LABEL_H, STRIP_H = 64, 40, 120
BAR_H = 14
Y_LO, Y_HI = -1.0, 1.25          # the strip's margin axis
KICK_FLASH_S = 0.3


def video_cfg(cfg, width: int, height: int) -> None:
  """The camera's frame size, only the rendered env's robot (no neighbours),
  and no debug drawings (command arrows, sensor rays)."""
  cfg.viewer.width, cfg.viewer.height = int(width), int(height)
  if hasattr(cfg.viewer, "max_extra_envs"):
    cfg.viewer.max_extra_envs = 0
  terms = list((getattr(cfg, "commands", None) or {}).values())
  sensors = getattr(cfg.scene, "sensors", None) or ()
  terms += list(sensors.values() if isinstance(sensors, dict) else sensors)
  for t in terms:
    if hasattr(t, "debug_vis"):
      t.debug_vis = False


def direction_name(rel: float) -> str:
  return DIRS_8[int(round(float(rel) / (math.pi / 4))) % 8]


def panel_label(name: str) -> str:
  latch = name.endswith("+latch")
  base = name[:-len("+latch")] if latch else name
  dwell = None
  if "+dwell" in base:
    base, dwell = base.split("+dwell", 1)
  p = base.split(":")
  if base == "sonic":
    s = "SONIC alone"
  elif base == "fallback":
    s = "SHAC alone"
  elif p[0] == "wmmin":
    s = f"WM tube {p[1]} s/{100 * float(p[2]):g}%"
  elif p[0] == "wmrollout":
    s = f"WM filter {p[1]} s (no tube)"
  elif p[0] == "rollout":
    s = f"exact-sim filter {p[1]} s"
  elif p[0] == "wmtube":
    s = f"WM per-step tube {p[1]} s"
  else:
    s = name
  # +dwell<s>: SHAC holds at least <s> after each hand-over ("latch <s>"); +latch: for good
  return s + (", latch forever" if latch else "") + (f", latch {dwell} s" if dwell else "")


def _font(size: int):
  from PIL import ImageFont
  try:
    from matplotlib import font_manager
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
  except Exception:
    return ImageFont.load_default(size=size)


class FilterVideo:
  """Frames and per-step traces of ``episodes`` under each hooked condition
  (``hook``), then the videos (``write``)."""

  def __init__(self, env, episodes, out_dir: str, eps: dict, categories, clips, dt: float,
               window: int, size=(480, 360), fps: int = 50, hold_s: float = 1.5,
               cols: int = 0):
    self.env, self.mj = env, env.mj
    self.sel = sorted({int(e) for e in episodes})
    self.out_dir, self.eps, self.cats, self.clips = out_dir, eps, list(categories), list(clips)
    self.dt, self.window, self.size, self.fps = float(dt), int(window), tuple(size), int(fps)
    self.hold = int(round(hold_s * fps))
    self.cols = int(cols)               # panels per row (0: one row)
    self.data: dict = {}
    self.big, self.mid, self.small = _font(26), _font(19), _font(16)

  # --- recording ------------------------------------------------------------------------

  def _point(self, slot: int) -> None:
    r = getattr(self.mj, "_offline_renderer", None)
    if r is not None and hasattr(r, "_cfg"):
      r._cfg.env_idx = int(slot)
    self.mj.cfg.viewer.env_idx = int(slot)

  def _frame(self, d: dict, slot: int) -> None:
    from PIL import Image
    self._point(slot)
    img = np.asarray(self.env.render())
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="JPEG", quality=88)
    d["frames"].append(buf.getvalue())

  def hook(self, cond: str, ev):
    """``Evaluator.run``'s ``on_step`` for condition ``cond``: renders the
    chosen episodes' slots and books ``ev.trace`` for them."""
    active: dict = {}

    def on_step(ep, k, dones):
      if dones is None:                       # a wave's reset: the spawns
        active.clear()
        epc = ep.detach().cpu()
        for e in self.sel:
          hit = (epc == e).nonzero(as_tuple=False)
          if hit.numel():
            slot = int(hit[0, 0])
            active[slot] = e
            d = self.data[(cond, e)] = {"frames": [], "eng": [], "verdict": [], "g": [],
                                        "slot": slot}
            self._frame(d, slot)
        return
      if not active:
        return
      tr = ev.trace
      pos = {s: i for i, s in enumerate(tr["ids"].tolist())}
      dn = dones.detach().cpu()
      for slot, e in list(active.items()):
        i = pos.get(slot)
        if i is None:                         # its episode ended on the step before
          del active[slot]
          continue
        d = self.data[(cond, e)]
        d["eng"].append(bool(tr["engaged"][i]))
        v = tr["verdict"]
        d["verdict"].append(math.nan if v is None else float(v[i]))
        d["g"].append(float(tr["g"][slot]))
        if bool(dn[slot]):                    # fell: the env has respawned, no frame
          del active[slot]
          continue
        self._frame(d, slot)

    return on_step

  # --- drawing --------------------------------------------------------------------------

  def _strip(self, d: dict, on: int) -> "Image.Image":
    """The full strip of one panel (every step); ``write`` reveals it step by step."""
    from PIL import Image, ImageDraw
    W = self.size[0]
    T = on + self.window
    im = Image.new("RGB", (W, STRIP_H), (22, 22, 26))
    dr = ImageDraw.Draw(im)
    x0, x1 = 8, W - 8
    y_top, y_bot = BAR_H + 6, STRIP_H - 6

    def X(step: float) -> float:
      return x0 + (x1 - x0) * step / T

    def Y(g: float) -> float:
      g = min(max(g, Y_LO), Y_HI)
      return y_bot - (y_bot - y_top) * (g - Y_LO) / (Y_HI - Y_LO)

    for j, e in enumerate(d["eng"]):       # decision j acts over [j, j + 1)
      dr.rectangle((X(j), 2, max(X(j + 1) - 0.5, X(j)), 2 + BAR_H), fill=GREEN if e else BLUE)
    for g in (1.0, 0.5):
      dr.line((x0, Y(g), x1, Y(g)), fill=(55, 55, 62), width=1)
    for xs in range(int(x0), int(x1), 8):
      dr.line((xs, Y(0.0), xs + 4, Y(0.0)), fill=RED, width=1)
    dr.line((X(on), y_top, X(on), y_bot), fill=(120, 85, 35), width=1)     # the kick

    def poly(vals, col, width):
      pts = []
      for j, v in enumerate(vals):
        if math.isfinite(v):
          pts.append((X(j + 1), Y(v)))
        elif len(pts) > 1:
          dr.line(pts, fill=col, width=width)
          pts = []
        else:
          pts = []
      if len(pts) > 1:
        dr.line(pts, fill=col, width=width)
      elif len(pts) == 1:
        dr.ellipse((pts[0][0] - 1, pts[0][1] - 1, pts[0][0] + 1, pts[0][1] + 1), fill=col)

    poly(d["verdict"], ORANGE, 2)
    poly(d["g"], WHITE, 2)
    dr.text((x0 + 2, Y(1.0) - 14), "1", font=self.small, fill=GREY)
    dr.text((x0 + 2, Y(0.0) + 1), "0", font=self.small, fill=RED)
    return im

  def _panel(self, img: np.ndarray, label: str, t: int, on: int, kick: str, acting,
             outcome) -> np.ndarray:
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    dr.rectangle((0, 0, w, LABEL_H), fill=(0, 0, 0, 160))
    dr.text((10, 9), label, font=self.mid, fill=WHITE)
    rel = (t - on) * self.dt
    clock = f"kick in {-rel:.2f} s" if t < on else f"kick +{rel:.2f} s"
    dr.text((w - 10, LABEL_H + 6), clock, font=self.small, fill=WHITE, anchor="ra")
    if acting is not None and outcome is None:
      txt, col = ("SHAC acting", GREEN) if acting else ("SONIC acting", BLUE)
      box = dr.textbbox((10, LABEL_H + 6), txt, font=self.small)
      dr.rectangle((box[0] - 5, box[1] - 4, box[2] + 5, box[3] + 4), fill=col + (230,))
      dr.text((10, LABEL_H + 6), txt, font=self.small, fill=WHITE)
    if on <= t < on + int(round(KICK_FLASH_S / self.dt)) and outcome is None:
      box = dr.textbbox((w // 2, h - 34), kick, font=self.mid, anchor="mm")
      dr.rectangle((box[0] - 12, box[1] - 7, box[2] + 12, box[3] + 7), fill=ORANGE + (230,))
      dr.text((w // 2, h - 34), kick, font=self.mid, fill=WHITE, anchor="mm")
    if outcome is not None:
      txt, col = outcome
      box = dr.textbbox((w // 2, h // 2), txt, font=self.big, anchor="mm")
      dr.rectangle((box[0] - 14, box[1] - 10, box[2] + 14, box[3] + 10), fill=col + (225,))
      dr.text((w // 2, h // 2), txt, font=self.big, fill=WHITE, anchor="mm")
    return np.asarray(im)

  def _rows(self, width: int, lines) -> list:
    """``lines`` (each a list of phrases), each wrapped at phrase boundaries to ``width``."""
    from PIL import Image, ImageDraw
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    rows = []
    for parts in lines:
      cur = ""
      for ph in (parts if isinstance(parts, (list, tuple)) else [parts]):
        cand = ph if not cur else cur + ";  " + ph
        if cur and probe.textlength(cand, font=self.small) > width - 24:
          rows.append(cur)
          cur = ph
        else:
          cur = cand
      rows.append(cur)
    return rows

  def _banner(self, width: int, lines, min_rows: int = 0) -> np.ndarray:
    """The banner over the panels: the wrapped ``lines``, at least ``min_rows``
    rows tall (every episode of a run gets the same height: one reel)."""
    from PIL import Image, ImageDraw
    rows = self._rows(width, lines)
    height = max(BANNER_H, 14 + 24 * max(len(rows), int(min_rows)))
    height += height % 2                                    # even, for yuv420p
    im = Image.new("RGB", (width, height), (25, 25, 30))
    dr = ImageDraw.Draw(im)
    for j, ln in enumerate(rows):
      dr.text((12, 7 + 24 * j), ln, font=self.small, fill=(235, 235, 235))
    return np.asarray(im)

  def _grid(self, tiles) -> np.ndarray:
    """``tiles`` (equal size) in rows of ``cols`` (all in one row when 0), a
    short last row padded with blank tiles."""
    n = len(tiles)
    c = n if self.cols <= 0 else min(self.cols, n)
    blank = np.full_like(tiles[0], 22)
    rows = []
    for r0 in range(0, n, c):
      row = list(tiles[r0:r0 + c])
      row += [blank] * (c - len(row))
      rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)

  # --- writing --------------------------------------------------------------------------

  def _episode_text(self, e: int):
    """(category, clip, |dv|, direction, kick label, banner lines) of episode ``e``."""
    on = int(self.eps["on"][e])
    cat = self.cats[int(self.eps["category"][e])] if self.cats else str(int(self.eps["category"][e]))
    ci = int(self.eps["clip"][e])
    clip = self.clips[ci] if ci < len(self.clips) else str(ci)
    dv, way = float(self.eps["dv"][e]), direction_name(float(self.eps["rel"][e]))
    kick = f"KICK {dv:.1f} m/s {way}"
    lines = [[f"episode {e}: {clip} frame {int(self.eps['frame0'][e])} ({cat})",
              f"kick |dv| = {dv:.2f} m/s toward the robot's {way}, {on * self.dt:.2f} s after "
              f"the spawn", "no sensor noise"],
             ["strip: bar = who acts (blue SONIC, green SHAC)", "white = real margin g",
              "orange = filter's verdict (predicted worst margin - eps; < 0 hands the step to "
              "SHAC)", "red dashes = 0"]]
    return cat, clip, dv, way, kick, lines

  def write(self, panels, recs: dict, meta: dict) -> None:
    import imageio
    from PIL import Image
    os.makedirs(self.out_dir, exist_ok=True)
    W, H = self.size
    on_all = self.eps["on"]
    paths, summary = [], []
    ncol_all = len(panels) if self.cols <= 0 else min(self.cols, len(panels))
    # one banner height for the whole run, so every episode's frames have one size
    min_rows = max([len(self._rows(W * ncol_all, self._episode_text(e)[5])) for e in self.sel]
                   or [0])
    for e in self.sel:
      runs = []
      on = int(on_all[e])
      for name in panels:
        d = self.data.get((name, e))
        if d is None or not d["frames"]:
          print(f"[video] episode {e}: nothing recorded under {name}", flush=True)
          continue
        rec = recs[name]
        surv, ttf = bool(rec["survived"][e]), float(rec["time_to_fall"][e])
        outcome = ((f"SURVIVED {self.window * self.dt:g} s", GREEN) if surv else
                   (f"FELL {ttf:.2f} s after the kick", RED))
        eng = np.asarray(d["eng"], dtype=bool)
        post = eng[on:] if eng.size > on else eng[:0]
        first = np.flatnonzero(eng)
        runs.append({"name": name, "label": panel_label(name), "d": d, "outcome": outcome,
                     "survived": surv, "ttf_s": None if surv else ttf,
                     "strip": self._strip(d, on),
                     "first_engaged_s": (float((first[0] - on) * self.dt) if first.size else None),
                     "engaged_post": float(post.mean()) if post.size else 0.0,
                     "switches": int((eng[1:] != eng[:-1]).sum() + (eng[:1].sum()))})
      if not runs:
        continue
      cat, clip, dv, way, kick, lines = self._episode_text(e)
      ncol = len(runs) if self.cols <= 0 else min(self.cols, len(runs))
      banner = self._banner(W * ncol, lines, min_rows)
      letters = "".join("S" if r["survived"] else "F" for r in runs)
      fname = f"ep{e:03d}_{cat}_{dv:.1f}ms_{way}_{letters}.mp4"
      path = os.path.join(self.out_dir, fname)
      T = max(len(r["d"]["frames"]) for r in runs)
      cache: dict = {}
      writer = imageio.get_writer(path, fps=self.fps, codec="libx264", quality=6,
                                  macro_block_size=1)
      try:
        for t in range(T + self.hold):
          tiles = []
          for j, r in enumerate(runs):
            fr = r["d"]["frames"]
            last = len(fr) - 1
            i = min(t, last)
            key = (j, i)
            if key not in cache:
              if len(cache) > 4 * len(runs):
                cache.clear()
              cache[key] = np.asarray(Image.open(io.BytesIO(fr[i])).convert("RGB"))
            img = cache[key]
            if img.shape[:2] != (H, W):
              img = np.asarray(Image.fromarray(img).resize((W, H)))
            eng = r["d"]["eng"]
            acting = eng[i - 1] if 1 <= i <= len(eng) else None
            done = t >= last and t > 0
            tile = self._panel(img, r["label"], i, on, kick, acting,
                               r["outcome"] if done else None)
            # the strip, revealed up to the steps taken so far
            n_steps = min(len(eng), max(i, 0)) if not done else len(eng)
            st = Image.new("RGB", (W, STRIP_H), (22, 22, 26))
            cut = int(round(8 + (W - 16) * n_steps / (on + self.window))) + 1
            st.paste(r["strip"].crop((0, 0, max(cut, 1), STRIP_H)), (0, 0))
            tiles.append(np.concatenate([tile, np.asarray(st)], axis=0))
          frame = np.concatenate([banner, self._grid(tiles)], axis=0)
          writer.append_data(frame)
      finally:
        writer.close()
      paths.append(path)
      row = {"episode": e, "file": fname, "clip": clip, "frame": int(self.eps["frame0"][e]),
             "category": cat, "dv": dv, "direction": way, "onset_s": on * self.dt,
             "panels": {r["name"]: {k: r[k] for k in ("label", "survived", "ttf_s",
                                                        "first_engaged_s", "engaged_post",
                                                        "switches")} for r in runs}}
      summary.append(row)
      print(f"[video] {fname}: " + "; ".join(
        f"{r['label']} {r['outcome'][0]} (SHAC {r['engaged_post']:.0%} of post-kick steps, "
        f"{r['switches']} hand-overs)" for r in runs), flush=True)
    if paths:
      writer = imageio.get_writer(os.path.join(self.out_dir, "reel.mp4"), fps=self.fps,
                                  codec="libx264", quality=6, macro_block_size=1)
      try:
        size = [0, 0]
        for p in paths:
          w_, h_ = imageio.get_reader(p).get_meta_data()["size"]
          size = [max(size[0], w_), max(size[1], h_)]
        for p in paths:
          for frame in imageio.get_reader(p):
            if frame.shape[0] != size[1] or frame.shape[1] != size[0]:   # pad to one size
              pad = np.full((size[1], size[0], 3), 22, dtype=frame.dtype)
              pad[:frame.shape[0], :frame.shape[1]] = frame
              frame = pad
            writer.append_data(frame)
      finally:
        writer.close()
    with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
      json.dump({"panels": list(panels), "policy": meta.get("policy"), "tag": meta.get("tag"),
                 "episodes": summary}, f, indent=1)
    with open(os.path.join(self.out_dir, "summary.txt"), "w") as f:
      for row in summary:
        f.write(f"{row['file']}: {row['clip']} frame {row['frame']} ({row['category']}), kick "
                f"{row['dv']:.2f} m/s {row['direction']} at {row['onset_s']:.2f} s\n")
        for name, p in row["panels"].items():
          out = "survived" if p["survived"] else f"fell {p['ttf_s']:.2f} s after the kick"
          fe = "never" if p["first_engaged_s"] is None else f"{p['first_engaged_s']:+.2f} s"
          f.write(f"    {p['label']:38s} {out:28s} first hand-over {fe:>8s}, SHAC on "
                  f"{p['engaged_post']:.0%} of post-kick steps, {p['switches']} hand-overs\n")
    print(f"[video] wrote {len(paths)} episodes, reel.mp4 and summary.txt to {self.out_dir}",
          flush=True)
