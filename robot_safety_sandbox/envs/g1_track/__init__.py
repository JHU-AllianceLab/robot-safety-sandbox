"""G1 squat tracking: a LAFAN segment as a task, in a thermal fallback's world.

``reference`` the segment: 50 fps frames, clipped joints, sysID forward
            kinematics, grounding, velocities; start-frame and placement math.
``command`` the "ref" command term: the reference per env, the spawn.
``mdp``     observation / reward / termination terms and the episode-stats bus.
``env_cfg`` the cfg builder: g1_recover_thermal{70,_cbf} with the task added.
"""
