# Ultracar
## Ai Automated vehicule

* First step is to set the optimal way to acquire enough information and process them
* In the second step we'll look on the way to set an algorithm and improve it
* The final goal is to be able to complete one lap around the circuit suggest in the Udacity simulation.

## How to set it
* Install [Miniconda](https://conda.io/miniconda.html) to use the environment setting.
* Set the environment in Power Shell
```python
# Use TensorFlow without GPU
conda env create -f environments.yml car-behavorial-cloning
```
* Go to Anaconda prompt > activate car-behavorial-cloning
* Got to the right directory (cd " ")
* Launch the simulator in autonomous
* python drive.py model.h5 (to test your AI) or python model.py (to train your AI)

## Model obtained
* model.h5 : their trained model
* model-000.h5 : our trained model with their utils for 1epoch
* model-001.h5 : our trained model with their utils for 6epoch
* ...

## Contributor

- Puissant Baeyens Victor, 12098, [MisterTarock](https://github.com/MisterTarock)
- De Keyzer  Paolo, 13201, [TouchTheFishy](https://github.com/TouchTheFishy)


# References

- Our "Artificial Intelligence" course
- The self driving simulator from Udacity https://github.com/udacity/self-driving-car-sim
