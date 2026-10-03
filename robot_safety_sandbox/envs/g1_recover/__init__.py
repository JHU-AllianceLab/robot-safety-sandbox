"""G1 fall-recovery: avoid-only training from a harvested start-state bank.

``states``  the dataset schema, heading-frame conversion and the one restore
            function the harvester and the training spawn share.
``spawn``   the mjlab reset event (mixture spawn + last_action seeding) and the
            per-env, per-source difficulty curriculum with its logging.
``env_cfg`` the cfg builder: g1_stabilize with the spawn swapped in.
"""
