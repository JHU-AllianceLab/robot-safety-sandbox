"""onnxruntime check of examples/export_fast_beatit_onnx.py's graphs: parity against the torch outputs recorded in
parity.npz, and batch-1 latency on the CPU (1 and 4 intra-op threads), the robot's likely runtime.

  <venv with onnxruntime>/bin/python examples/fast_beatit_onnx_check.py --dir <export folder> [--n 400 --reps 300]

Writes <dir>/onnx_check.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import time

import numpy as np
import onnxruntime as ort


def session(path: str, threads: int):
  so = ort.SessionOptions()
  so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  so.intra_op_num_threads = threads
  so.inter_op_num_threads = 1
  so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
  return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--dir", required=True)
  p.add_argument("--n", type=int, default=400, help="parity samples")
  p.add_argument("--reps", type=int, default=300)
  p.add_argument("--threads", type=int, nargs="+", default=[1, 4])
  a = p.parse_args()
  P = np.load(os.path.join(a.dir, "parity.npz"))
  flat = lambda x: x.reshape(-1, *x.shape[2:]) if x.ndim >= 2 else x.reshape(-1)  # noqa: E731
  S, C, A, F = (flat(P[k]) for k in ("fb_state", "context", "proposal", "fb_obs"))
  rng = np.random.default_rng(0)
  idx = rng.choice(len(S), size=min(a.n, len(S)), replace=False)
  res = {"cpu": platform.processor() or platform.machine(), "ort": ort.__version__, "graphs": {}}
  try:
    res["cpu_model"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
  except Exception:  # noqa: BLE001
    pass
  for path in sorted(glob.glob(os.path.join(a.dir, "*.onnx"))):
    name = os.path.basename(path)
    g = {"MB": os.path.getsize(path) / 1e6}
    sess = session(path, 1)
    if name == "beatit_fallback.onnx":
      out = np.stack([sess.run(None, {"fb_obs": F[i:i + 1]})[0][0] for i in idx])
      g["parity_max_abs"] = float(np.abs(out - flat(P["a_fallback"])[idx]).max())
      feed = {"fb_obs": F[:1]}
    else:
      feed_of = lambda i: {"fb_state": S[i:i + 1], "context": C[i:i + 1], "proposal": A[i:i + 1]}  # noqa: E731
      out = np.array([sess.run(None, feed_of(i))[0][0] for i in idx])
      ref = flat(P["margin_min"]) if name == "beatit_monitor_full.onnx" else \
        flat(P[name.replace("beatit_monitor_", "").replace(".onnx", "") + "_sigma"])
      d = np.abs(out - ref[idx])
      g["parity_max_abs"] = float(d.max())
      g["parity_p99_abs"] = float(np.quantile(d, 0.99))
      feed = feed_of(int(idx[0]))
    for th in a.threads:
      s = session(path, th)
      for _ in range(20):
        s.run(None, feed)
      ts = []
      for _ in range(a.reps):
        t0 = time.perf_counter()
        s.run(None, feed)
        ts.append(time.perf_counter() - t0)
      ts = np.array(ts) * 1e3
      g[f"ms_threads{th}"] = {"median": float(np.median(ts)), "p99": float(np.quantile(ts, 0.99)),
                              "max": float(ts.max())}
    res["graphs"][name] = g
    print(f"[ort] {name:28s} {g['MB']:6.1f} MB parity max {g['parity_max_abs']:.2e} | "
          + " | ".join(f"{th} thr {g[f'ms_threads{th}']['median']:.2f} ms (p99 {g[f'ms_threads{th}']['p99']:.2f})"
                       for th in a.threads), flush=True)
  json.dump(res, open(os.path.join(a.dir, "onnx_check.json"), "w"), indent=1)
  print(f"[ort] wrote {a.dir}/onnx_check.json ({res.get('cpu_model', res['cpu'])})")


if __name__ == "__main__":
  main()
