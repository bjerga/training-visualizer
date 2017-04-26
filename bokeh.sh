#!/usr/bin/env bash
bokeh serve --allow-websocket-origin=localhost:5000 \
	custom_bokeh/training_progress.py \
	custom_bokeh/layer_activations.py \
	custom_bokeh/saliency_maps.py