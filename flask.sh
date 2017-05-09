#!/usr/bin/env bash
export FLASK_APP=visualizer
export PYTHONPATH=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
python3 -m flask run