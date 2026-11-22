#!/bin/bash
# Start the RealSense frame server as root (macOS password dialog), once per session.
# Root cannot read ~/Documents (TCC), so the server runs from ~/realsense_tools with its own venv.
# Extra arguments go to rs_server.py, e.g.:  scripts/real/start_camera.sh --fps 60 --exposure 3000
# Stop it with:  .venv/bin/python -c "from eaiproject.realsense_client import RealSenseClient as C; C().stop_server()"
set -e
T="$HOME/realsense_tools"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$T"
cp "$HERE/rs_server.py" "$T/rs_server.py"
if [ ! -x "$T/.venv/bin/python" ]; then
  "$HOME/.local/bin/uv" venv --python 3.12 "$T/.venv" -q
  "$HOME/.local/bin/uv" pip install -q --python "$T/.venv/bin/python" numpy "pyrealsense2-macosx==2.54.2"
fi
if nc -z 127.0.0.1 8770 2>/dev/null; then echo "camera server already running"; exit 0; fi
ARGS="$*"
osascript -e "do shell script \"[ -f $T/rs_server.pid ] && kill -9 \$(cat $T/rs_server.pid) 2>/dev/null; sleep 1; cd $T && .venv/bin/python rs_server.py $ARGS < /dev/null > $T/rs_server.log 2>&1 &\" with administrator privileges"
for i in $(seq 1 40); do
  if nc -z 127.0.0.1 8770 2>/dev/null; then cat "$T/rs_server.log"; exit 0; fi
  sleep 0.25
done
echo "server did not start; log:"; cat "$T/rs_server.log"; exit 1
