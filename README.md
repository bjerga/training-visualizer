# training-visualizer
A website application for visualizing the training progress of neural networks. Application is meant to run on a Linux machine.

## Installation
```
pip3 install -r visualizer/requirements.txt
```

## Running
To create a database and run the application, execute:
```
export FLASK_APP=visualizer
python3 -m flask initdb
python3 -m flask run
```

If a database is already created, run the application by executing:
```
export FLASK_APP=visualizer
python3 -m flask run
```

For Windows you need to use 'set' instead of 'export'. 
If computer only has Python 3.x installed, use 'python' instead of 'python3'. Similar code alterations may need to be applied.
Mac OS machines cannot run application due to error when plotting outside main process. 
