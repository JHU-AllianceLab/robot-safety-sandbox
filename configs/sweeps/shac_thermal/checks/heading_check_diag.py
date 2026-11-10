"""Is SonicReference.build's orientation check ill-conditioned at lying-down frames?

Resamples the train library's clips on mjlab 1.2's grid (what the library was
built on), then evaluates the check (1 - |<q_h, q_lib>|, q_h = the yaw-free
root quaternion) on CPU float32, CPU float64 and, if there is one, CUDA float32;
and a yaw-free alternative, the gravity direction in the root frame.
"""
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from mjlab.utils.lab_api.math import quat_apply_inverse  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.sonic import _motion_loader  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.states import load_bank, to_heading_frame  # noqa: E402

lib = load_bank("data/g1_lafan1_clips_train.pt", "cpu")
m = lib["meta"]
quats, lengths = [], []
for name in m["clips"]:
  ml = _motion_loader(os.path.join("data/lafan1_g1", f"{name}.csv"), 30, 50, "mjlab-1.2")
  quats.append(ml.motion_base_rots)
  lengths.append(int(ml.output_frames))
assert lengths == list(m["clip_frames"])
root_q = torch.cat(quats).float()
start = torch.cumsum(torch.tensor(lengths), 0) - torch.tensor(lengths)
rows = start[lib["clip"].long()] + lib["frame"].long()
lib_q = lib["root_quat_h"].float()
g = torch.tensor([0.0, 0.0, -1.0])


def check(q, lq, dev, dtype):
  q, lq = q.to(dev, dtype), lq.to(dev, dtype)
  z = torch.zeros(q.shape[0], 3, device=dev, dtype=dtype)
  qh = to_heading_frame(q, z, z)[0]
  err = 1.0 - (qh * lq).sum(1).abs()
  gv = g.to(dev, dtype).expand(q.shape[0], 3)
  pg = (quat_apply_inverse(q, gv) - quat_apply_inverse(lq, gv)).norm(dim=1)
  return err.cpu().double(), pg.cpu().double()


q = root_q[rows]
up = -quat_apply_inverse(q, g.expand(q.shape[0], 3))[:, 2]       # cos(tilt)
tilt = torch.rad2deg(torch.acos(up.clamp(-1, 1)))
for dev, dtype in [("cpu", torch.float32), ("cpu", torch.float64)] + (
    [("cuda", torch.float32)] if torch.cuda.is_available() else []):
  err, pg = check(q, lib_q, dev, dtype)
  i = int(err.argmax())
  big = err > 1e-4
  print(f"[diag] {dev} {dtype}: max 1-|<q,q'>| {float(err.max()):.2e} (row {i}, tilt "
        f"{float(tilt[i]):.1f} deg, clip {m['clips'][int(lib['clip'][i])]}); rows > 1e-4: "
        f"{int(big.sum())} of {err.numel()}, their tilt "
        f"{float(tilt[big].min()) if big.any() else 0:.1f}-"
        f"{float(tilt[big].max()) if big.any() else 0:.1f} deg | gravity-direction diff max "
        f"{float(pg.max()):.2e}, p99.99 {float(torch.quantile(pg.float(), 0.9999)):.2e}",
        flush=True)
