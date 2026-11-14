"""Adversarial Motion Priors (Peng et al. 2021) for the squat-tracking task policy.

``features``      the per-frame feature phi(s) shared by the env term and the expert
                  builder (pure torch: no mjlab, no sandbox import)
``discriminator`` D(phi_t, phi_t+1), its LSGAN loss, the style reward, the expert
                  pairs and the policy-pair replay
``env``           AMPTaskEnv: mixes the style reward into the task reward the PPO
                  collector sees, and gathers the discriminator's policy pairs
``ppo``           AMPPPO (safety_sb3's CumulativePPO1P + the discriminator update)
                  and the checkpoint callback for D

Nothing here is imported by this ``__init__``: ``features`` must stay importable
from an mjlab observation term without pulling in safety_sb3.
"""
