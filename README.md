# training-visualizer
A web interface for visualizing data produced while training artificial neural networks.

The example network `mnist_keras.py` along with example visualization input images are available in the `test_code` folder.

## Installation & Setup
This explains how to install and set up the visualization tool ready for use. We assume that the user have Python 3.5 installed.

### Installation
The first thing you need to do is download or clone the GitHub repository to your computer. Open a command line interface and navigate to the main directory of the repository, `training-visualizer`.

Install all required packages by running the following command in the command line interface:

```
pip install -r visualizer/requirements.txt
```

**Note:** If you already have an installation of the Python Imaging Library (PIL) package, you will get a failure when trying to install the Pillow package listed in the requirements file. Remove Pillow from the requirements file and rerun the command above.

### Configuration and Setup
The visualizer folder of the project contains a configuration file that can be used to configure some settings for the application, most of which are not of importance for a typical user. However, if you are using a different command than `python3` to run Python programs, it is crucial that you change the `PYTHON` variable to the actual command you are using.

After the application has been properly configured, we can start performing the necessary setup. To make this process easier for the user, we have created bash scripts for Linux and MacOS, and batch scripts for Windows, containing the necessary commands.

#### Linux and MacOS

For Linux and MacOS, the only thing we need to set up the visualization tool is initializing the database. The environment setup is included in the scripts for starting the application, which we will see in a later section. The command for initializing the database is as follows:

```
source linux_macos/init_db.sh
```

#### Windows

For Windows, we need to both set up the environment and initialize the database. This is done by running these two batch files in the command line:

```
windows/init_env.bat
windows/init_db.bat
```

### Starting the Visualization Tool

The application consists of two separate processes in order to function correctly: a Flask application server, and a Bokeh visualization server. We will now demonstrate how to start these servers.

#### Linux and MacOS

In the same command line window that you executed the database setup command, run the following in order to start the Flask application server:

```
source linux_macos/start_flask.sh
```

Open a new command line window, navigate to the same folder, and then run this command to start the Bokeh visualization server:

```
source linux_macos/start_bokeh.sh
```

#### Windows

In the same command line window that you executed the environment and database setup commands, run the following batch file in order to start both the Flask application server and the Bokeh visualization server:

```
windows/start_servers.bat
```

Navigate to [localhost:5000](localhost:5000) in a web browser (we recommend Google Chrome), and the visualization tool login page should be displayed.