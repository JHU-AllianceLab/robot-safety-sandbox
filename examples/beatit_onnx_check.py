"""Check an exported Beat-It filter folder with numpy + onnxruntime only (e.g. on the robot's computer):

    python check_onnx.py [--folder .] [--threads 1] [--rows 800]

1. both graphs against the PyTorch export on the recorded sim run (parity.npz) and against the
   sandbox's calibrated monitor (ref_margin_min / ref_survival);
2. the accept / reject decision at spec.json's threshold (or the tube eps when none is given);
3. time per call of each graph (the control step is 20 ms).
Ends with ALL OK; writes check.json beside the graphs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--folder", default=str(Path(__file__).resolve().parent))
  p.add_argument("--threads", type=int, default=1)
  p.add_argument("--rows", type=int, default=800)
  p.add_argument("--timing-calls", type=int, default=200)
  a = p.parse_args()
  import onnxruntime as ort

  d = Path(a.folder)
  spec = json.loads((d / "spec.json").read_text())
  R = np.load(d / "parity.npz")
  so = ort.SessionOptions()
  so.intra_op_num_threads = a.threads
  so.inter_op_num_threads = 1
  prov = ["CPUExecutionProvider"]
  fb = ort.InferenceSession(str(d / "beatit_fallback.onnx"), so, providers=prov)
  mon = ort.InferenceSession(str(d / "beatit_monitor.onnx"), so, providers=prov)

  flat = lambda k: R[k].reshape(-1, *R[k].shape[2:]).astype(np.float32)  # noqa: E731
  X = {k: flat(k) for k in ("fb_obs", "fb_state", "context", "proposal", "a_fallback", "margin_min",
                            "survival", "tw_max_c", "ref_margin_min", "ref_survival")}
  n = X["fb_obs"].shape[0]
  rows = np.linspace(0, n - 1, min(a.rows, n)).astype(int)
  gz = spec.get("upright_col")
  upright = (X["fb_obs"][:, gz] < -0.8) if gz is not None else np.ones(n, bool)
  thr = spec["monitor"]["threshold"]
  thr = spec["monitor"]["tube_eps_1s_95"] if thr is None else thr

  e = {"fallback": [], "margin_vs_torch": [], "margin_vs_ref": [], "survival_vs_ref": [], "tw_vs_torch": []}
  flips, near = 0, 0
  for i in rows:
    act = fb.run(None, {"fb_obs": X["fb_obs"][i:i + 1]})[0]
    mm, sv, tw = mon.run(None, {"fb_state": X["fb_state"][i:i + 1], "context": X["context"][i:i + 1],
                                "proposal": X["proposal"][i:i + 1]})
    if not upright[i]:
      continue
    e["fallback"].append(np.abs(act[0] - X["a_fallback"][i]).max())
    e["margin_vs_torch"].append(abs(mm[0] - X["margin_min"][i]))
    e["margin_vs_ref"].append(abs(mm[0] - X["ref_margin_min"][i]))
    e["survival_vs_ref"].append(abs(sv[0] - X["ref_survival"][i]))
    e["tw_vs_torch"].append(np.abs(tw[0] - X["tw_max_c"][i]).max())
    if (mm[0] >= thr) != (X["ref_margin_min"][i] >= thr):
      flips += 1
      near += abs(X["ref_margin_min"][i] - thr) < 0.01
  m = {k: float(np.max(v)) if v else float("nan") for k, v in e.items()}
  n_up = int(upright[rows].sum())
  print(f"rows checked {len(rows)} (upright {n_up}); max abs on upright states:")
  for k, v in m.items():
    print(f"  {k:18s} {v:.2e}")
  print(f"  decisions at threshold {thr:.4f}: {flips} flips of {n_up} ({near} within 0.01 of it)")
  tol = {"fallback": 1e-4, "margin_vs_torch": 1e-2, "margin_vs_ref": 1e-2, "survival_vs_ref": 1e-2,
         "tw_vs_torch": 0.5}
  ok = all(m[k] <= tol[k] for k in tol) and flips <= max(1, n_up // 200)

  i = int(rows[len(rows) // 2])
  t = {}
  for name, sess, feed in (("fallback", fb, {"fb_obs": X["fb_obs"][i:i + 1]}),
                           ("monitor", mon, {"fb_state": X["fb_state"][i:i + 1],
                                             "context": X["context"][i:i + 1],
                                             "proposal": X["proposal"][i:i + 1]})):
    for _ in range(10):
      sess.run(None, feed)
    ts = []
    for _ in range(a.timing_calls):
      t0 = time.perf_counter()
      sess.run(None, feed)
      ts.append(1e3 * (time.perf_counter() - t0))
    t[name] = {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}
    print(f"  {name:9s} {t[name]['median_ms']:.2f} ms median, {t[name]['p95_ms']:.2f} ms p95 "
          f"({a.threads} CPU thread{'s' if a.threads > 1 else ''}, {ort.get_device()})")
  (d / f"check_t{a.threads}.json").write_text(json.dumps(
    {"max_abs_upright": m, "tolerances": tol, "decision_flips": flips, "rows": len(rows), "upright": n_up,
     "threshold": thr, "timing": t, "threads": a.threads, "onnxruntime": ort.__version__, "ok": ok}, indent=1))
  print("ALL OK" if ok else "CHECK FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
