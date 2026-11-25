# config

`rig_map.json` is written by `scripts/real/rig_solve.py` and committed: the corner positions of the
table markers in the arm's kinematic base frame, the table plane, and the fit statistics. It holds
no images. `scripts/real/relocalize.py` reads it to recover the camera pose each session.

Redo the static calibration ([docs/calibration_protocol.md](../docs/calibration_protocol.md)) and
commit the new file whenever a marker, the arm or the table has moved.
