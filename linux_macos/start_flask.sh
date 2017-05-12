#!/usr/bin/env bash
current_dir="$( dirname "${BASH_SOURCE[0]}" )"
export FLASK_APP=visualizer
export PYTHONPATH=$( cd "$( dirname "$current_dir" )" && pwd )
python3 -m flask run