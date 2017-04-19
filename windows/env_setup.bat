@echo off
set FLASK_APP=visualizer
echo Flask app set to visualizer
set current_dir=%~dp0
set PYTHONPATH=%PYTHONPATH%;%current_dir:~0,-8%
echo PYTHONPATH set to %PYTHONPATH%