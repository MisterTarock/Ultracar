# Ultracar
## Ai Automated vehicule goal

* First step is to set the optimal way to acquire enough information and process them
* In the second step we'll look on the way to set an algorithm and improve it
* The final goal is to be able to complete one lap around the circuit suggest in the Udacity simulation.

## How to generate some data before training
* create a folder "data" in the Ultracar directory
* launch the simulator in training mode
* press "R" and select your "data" directory
* Play the game
* Save with "R" it will rerun your lap and save the .csv and IMG folder in your data directory


## How to set it
* Install [Miniconda](https://conda.io/miniconda.html) to use the environment setting.
* Set the environment in Power Shell
```python
# Use TensorFlow without GPU
conda env create -f environments.yml car-behavioral-cloning
```
* Go to Anaconda prompt > activate car-behavioral-cloning
* Got to the right directory (cd " ")
* Launch the simulator in autonomous
* python drive.py model.h5 (to test your AI) or python model.py (to train your AI)

## Model obtained
* model.h5 : their trained model
* model-000.h5 : our trained model with their utils for 1epoch
* model-001.h5 : our trained model with their utils for 6epoch
* ...

## Files description
* .gitignore : allow us to not sync our IMG folder (it's quite heavy) and our .csv file (it contains the path variable for the image so it's not very shareable).
* ECAM-AI-Project.pdf : it's the file with all the instructions for this lab.
* drive.py : It's the link between the simulator and our generated model.
* environments.yml : it's a list of dependencies that will automatically be installed inside the environment instead of a separated call of each of them with pip. (N.B. the environments-gpu.yml is also available to install tensorflow-gpu but we choose to note use it because of the difficulties around the requirements for a complete functioning tensorflow-gpu set-up)
Inside the environment file we wilol find this dependencies:
-




## Contributor

- Puissant Baeyens Victor, 12098, [MisterTarock](https://github.com/MisterTarock)
- De Keyzer  Paolo, 13201, [TouchTheFishy](https://github.com/TouchTheFishy)


# References

- Our "Artificial Intelligence" course
- The self driving simulator from Udacity https://github.com/udacity/self-driving-car-sim
