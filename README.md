# Prediction Kirby's Dream Land

STILL A WIP

## Project Overview
For this project, I build models taught in DA 460 - Predictive Analytics (at the Univeristy of North Alabama) to predict what move someone should make when playing Kirby's Dream Land at any given time. A description of the files found in this project are shown below:

| File/Folder             | Description                                                                      |
|-------------------------|----------------------------------------------------------------------------------|
| KirbysDreamLand-GameBoy | The integrated game KirbysDreamLand                                              |
| DA_460_Project.pdf              | The paper for the course project                                                        |
| DA_460_Project.zip            | LaTeX source of the paper |
| game_data.py                    | python file to create kdl.csv                |
| get_data.py                    | python file to data.json used in KirbysDreamLand-GameBoy                 |
| index.html                    | html file containing the decision tree, it can be viewed [here]()                |
| kdl_csv.zip                    | zip file that contains kdl.csv (the data used in the project)                |
| models.py                   | python program to build/test models                |
| params.py                   | python program to find parameters for the  models                |
| summary_statistics.py                    | python program to output summary statistics of the data                |

## Build Instructions

First, ensure python version 8.0 is installed, it can be installed [here](https://www.python.org/downloads/release/python-380/).

Next, go to where python is installed in the terminal/cmd prompt using 

```sh
cd <directory>
```

The location is different based on the operating system. For windows, it is ```C:\Users\<user>\AppData\Local\Programs\Python\Python38```, for linux run the following command to find the path:

```
which python38
```

Once in the directory, on the command prompt, run the following commands (on windows, replace python with python.exe):

```sh
python -m pip install --upgrade pip
python -m pip install gym-retro
python -m pip install gym==0.21
python -m pip install scikit-learn
python -m pip install matplotlib
python -m pip install pandas
python -m pip install keyboard
```

Next, in a separate window, move the KirbysDreamLand-GameBoy folder to the following directory: 
```sh
<directory of python installation>/Lib/site-packages/retro/data/stable/
```

Go back to the terminal/cmd window and run the following command, replace directory with the location of the KirbysDreamLand-GameBoy folder (on windows, replace python with python.exe):

```sh
python -m retro.import <directory>
```

Now, you can open idle (python version 8.0) and open->run project.py to execute the project. 
