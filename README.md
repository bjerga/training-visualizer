# training-visualizer
A website application for visualizing data produced while training an artificial neural networks.

## Installation
To install all requirement packages:
```
pip3 install -r visualizer/requirements.txt
```

## Setup
To initialize the application and set up a database.
```
export FLASK_APP=visualizer
python3 -m flask initdb
```
For Windows you need to use 'set' instead of 'export'. 

## Running
To run the application server, execute this in one terminal:
```
python3 -m flask run
```

Execute this in another terminal window or tab to run the visualization server:
```
bash bokeh.sh
```

This application assumes that Python 3.x is run from the terminal using 'python3'.
