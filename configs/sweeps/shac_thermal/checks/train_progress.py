"""Training progress of the thermal SHAC seeds: survival and why episodes end, per ~50M steps."""
import re
import sys

KEYS = {"surv": "Curriculum/recover/surv_handover", "fail": "fail_frac", "tmax": "Episode_Metrics/motor_temp_max",
        "fell": "Episode_Termination/fell_over", "contact": "Episode_Termination/illegal_co", "hot": "Episode_Termination/motor_over",
        "rest": "Episode_Termination/recovered", "tout": "Episode_Termination/time_out", "gmin": "g_min"}
for path in sys.argv[1:]:
  blocks, cur = [], {}
  for line in open(path, errors="ignore"):
    m = re.match(r"\|\s+(\S.*?)\s+\|\s+(\S+)\s+\|", line)
    if not m:
      continue
    k, v = m.group(1), m.group(2)
    if k == "total_timesteps":
      cur["steps"] = float(v)
      blocks.append(cur)
      cur = {}
      continue
    for short, full in KEYS.items():
      if k.startswith(full.split("/")[-1] if short == "gmin" else full):
        try:
          cur[short] = float(v)
        except ValueError:
          pass
  print(f"== {path.split('/')[-1]}: {len(blocks)} log blocks, at {blocks[-1]['steps'] / 1e6:.0f}M steps")
  print(f"{'steps':>7} {'surv':>5} {'fail':>5} | ends: {'fell':>5} {'contact':>7} {'OVERHEAT':>8} {'rest':>5} {'timeout':>7} | {'peak Tw':>7} {'img g_min':>9}")
  edges = list(range(0, int(blocks[-1]["steps"]) + 50_000_000, 50_000_000))
  for lo, hi in zip(edges[:-1], edges[1:]):
    w = [b for b in blocks if lo <= b["steps"] < hi]
    if not w:
      continue
    mean = lambda k: sum(b.get(k, 0.0) for b in w) / len(w)  # noqa: E731
    tot = sum(sum(b.get(k, 0.0) for k in ("fell", "contact", "hot", "rest", "tout")) for b in w) or 1.0
    sh = {k: sum(b.get(k, 0.0) for b in w) / tot for k in ("fell", "contact", "hot", "rest", "tout")}
    print(f"{hi / 1e6:6.0f}M {mean('surv'):5.3f} {mean('fail'):5.3f} | ends: {sh['fell']:5.3f} {sh['contact']:7.3f} "
          f"{sh['hot']:8.3f} {sh['rest']:5.3f} {sh['tout']:7.3f} | {mean('tmax'):7.1f} {mean('gmin'):9.3f}")
