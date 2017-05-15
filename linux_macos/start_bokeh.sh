#!/usr/bin/env bash
current_dir="$( dirname "${BASH_SOURCE[0]}" )"
export PYTHONPATH=$( cd "$( dirname "$current_dir" )" && pwd )
bokeh serve --allow-websocket-origin=localhost:5000 \
	custom_bokeh/training_progress.py \
	custom_bokeh/layer_activations.py \
	custom_bokeh/saliency_maps.py \
	custom_bokeh/deep_visualization.py \
	custom_bokeh/deconvolution_network.py