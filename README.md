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
Inside the environment file we will find this dependencies:

    | environments|car-behavioral-cloning|
    | ------ | ------ |
    |Python==3.5.2| The compatible version with TensorFlow 1.1 (warning: it's already the case with this environment but if you install python separatly make sure that it's the 64bit version to be compatible with tensorflow). |
    |numpy| Matrice and array processing |
    |matplotlib| Extension of NumPy for object-oriented API and plot generation |
    |jupyter| *unused* |
    |opencv3| Real-time computer vision |
    |pillow| Python Imaging Library, ad support for opening and saving the differents image after modification |
    |scikit-learn| library of diverses classifier to train our model |
    |scikit-image| image processing, *unused* |
    |scipy| math |
    |h5py| model... |
    |eventlet| server communication  |
    |flask-socketio| server creation|
    |seaborn| ? |
    |pandas| file processing (csv)|
    |imageio| image |
    |moviepy| ? |
    |tensorflow==1.1| |
    |keras==1.2|  |

The environments file isn't modified from the Sourcell version but all depedencies weren't used in the end.



## Contributor

- Puissant Baeyens Victor, 12098, [MisterTarock](https://github.com/MisterTarock)
- De Keyzer  Paolo, 13201, [TouchTheFishy](https://github.com/TouchTheFishy)


# References

- Our "Artificial Intelligence" course
- The self driving simulator from Udacity https://github.com/udacity/self-driving-car-sim
- The self driving car from Siraj Raval https://github.com/llSourcell/How_to_simulate_a_self_driving_car
