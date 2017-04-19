# training-visualizer
A website application for visualizing data produced while training artificial neural networks.

## Installation
To install all requirement packages:
```
pip3 install -r visualizer/requirements.txt
```

## Setup
To run the application as a whole, you need two terminal windows:
1. The flask application
2. The bokeh visualization server


To initialize the application and set up a database, execute these commands in the first window.
```
export FLASK_APP=visualizer
python3 -m flask initdb
```
To ensure that Python sees visualizer as a module, execute this command in both windows.

```
export PYTHONPATH="/path/to/visualizer/folder"
```

For Windows you need to use `set` instead of `export`.

**NOTE:**
This application assumes that Python 3.x is run from the terminal using `python3`. If you are using any other command is used to run python, you need to change this accordingly in the config file.

## Running
To run the flask application server, execute this in the first window.
```
python3 -m flask run
```

Execute this in the second window to run the bokeh visualization server.
```
bash bokeh.sh
```

Navigate to [localhost:5000](localhost:5000).