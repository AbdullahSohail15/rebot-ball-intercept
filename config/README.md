# config

`rig_map.json` is written by `scripts/real/calibrate.py` and committed: the corner positions of the
table markers in the arm's kinematic base frame, the table plane, and the fit statistics. It holds
no images. `real_tracking.py` and `check_calibration.py` read it to find the camera pose and the
table height.

Redo the calibration ([docs/calibration_protocol.md](../docs/calibration_protocol.md)) and
commit the new file whenever a marker, the arm or the table has moved.
