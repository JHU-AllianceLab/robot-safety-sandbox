"""G1 reach-avoid-stay to a waist-posture + body-velocity goal.

``command``     the per-episode waist (roll, pitch) goal and its magnitude
                curriculum hook.
``stay_value``  the frozen stay certificate V_stay the reach task's target is
                built from, and the noise-free rebuild of the current actor obs
                it is evaluated on.
``env_cfg``     the margins, the stay spawn and the two cfg builders (stay,
                reach).
"""
