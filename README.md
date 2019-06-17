# Ultracar
## Ai Automated vehicle goal

* First step is to set the optimal way to acquire enough information and process it
* In the second step we'll implement an algorithm and try improve it
* The final goal is to be able to complete one lap around the circuit suggests in the Udacity simulation.

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

## Usage

* Open the anaconda prompt and type

  ```python
  activate car-behavioral-cloning
  ```
* Got to the right directory

  ```python
  cd "path"
  ```
   *N.B. If you have to change disk storage to find your directory, just write this before (D is the letter of the wanted disk):*

   ```python
   D:
   ```

* Launch the simulator in autonomous
* Write one of the two following commands in the prompt

    ##### Test your model
    ```python
    python drive.py model.h5
    ```
    ##### Train your model
    ```python
    python model.py
    ```
* At the end of your training or test phase, exit with a simple *ctrl-c*.
*N.B. In the testing phase, you have to exit before shutting down the simulation or the prompt won't respond until restart of the simulation.*

## Model obtained

* model_final.h5 : Our final, working model.
* model-000.h5 : our trained model (first epoch)
* model-001.h5 : our trained model (second epoch)
* ...

The *epoch* is a specific time measure. The first epoch is the beginning of the training phase (the time zero) and the following ones are the different iterations of the training phase. During the training phase *model.py* will estimate the efficiency of the test model produced and then, if it had a better performance than the previous ones, save it with the epoch number in the file name.

For example, during the training phase:
* at the first iteration, it will overwrite the model-000 because it estimate that this time the model produced was better.
* at the second iteration, it won't save it because it estimate that the previous model-000 was better.
* at the third iteration, the loss is less than any other epoch so the model model-002.h5 will be saved.
* then, for all the following epochs, the loss will be never as low as the third epoch so no other model will be saved

For this example we will have the following files in the directory
* model-000.h5
* model-002.h5

## Files description

* .gitignore : allows us to ignore IMG folder (it's quite heavy) and our .csv file when versioning (it contains the path variable for the image so it's not very shareable).
* ECAM-AI-Project.pdf : file with all the instructions for this lab.
* drive.py : link between the simulator and our generated model.
* model-{xxx}.h5 : The different test models produced by our training phase. The number is the epoch number and indicates its iteration number.
* model_final.py : Our final, working model.
* utils.py : a set of processes applied on our images to assist *model.py*  
* environments.yml : it's a list of dependencies that will automatically be installed inside the environment instead of a separated call of each of them with pip. (N.B. the environments-gpu.yml is also available to install tensorflow-gpu but we choose to note use it because of the difficulties around the requirements for a complete functioning tensorflow-gpu set-up)
Inside the environment file we will find these dependencies:

    | environments|car-behavioral-cloning|
    | ------ | ------ |
    |Python==3.5.2| The compatible version with TensorFlow 1.1 (warning: it's already the case with this environment but if you install python separately make sure that it's the 64bit version to be compatible with tensorflow). |
    |numpy| Matrices and array processing |
    |matplotlib| Extension of NumPy for object-oriented API and plot generation |
    |opencv3| Real-time computer vision |
    |pillow| Python Imaging Library, ad support for opening and saving the different image after modification |
    |scikit-learn| Library of various classifier to train our model |
    |scipy| Another mathematical module for optimization, linear algebra,... to complete NumPy |
    |h5py| Used to generate the models, it's a set of file formats designed to store and organize large amount of data |
    |eventlet| In addition to SocketIO, allow a high concurrent networking to boost the performance of the client-server communication  |
    |flask-socketio| An equivalent of the classical socketIO by the Flask team, the purpose is the same, enable real-time communication between a server and his client|
    |pandas| Data manipulation for structured files as our *driving-log.csv*|
    |tensorflow==1.1| High performance numerical computation, allows us the set-up of the Neural Network  |
    |keras==1.2| It's user-friendly interface above TensorFlow with a set of simplified command to place *layers, activation functions, optimizers, ...*  |

The environments file isn't modified from the [||Source||](https://github.com/llSourcell/How_to_simulate_a_self_driving_car) version but some dependencies weren't used in the end (such as *seaborn, imageio, moviepy, scikit-image, jupyter*).

## Code explanation

### Model generation : model.py

This program is used to generate the model files that the car uses to drive itself.

When started, this program uses the following parameters :

* -d : Data directory [Default: "data" folder in the root]
* -t : When splitting the test batches, selects the proportion of the validation part [Default : 0.2]
* -k : Sets the dropout probability for the dropout layer of the NN [Default : 0.5]
* -n : Number of epochs (number of trainings) [Default :10]
* -s : Sets the amount of samples used per training [Default : 20000]
* -b : Sets the amount of images per batch [Default : 40]
* -o : Sets if only the best models are to be saved [Default: true]
* -l : Sets the learning rate for the NN [Default : 1e-4]

The code works the following way:

First, we load the training data:
```python
#reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    #we export the images (divided per camera) in a new variable X (input data)
    X = data_df[['center', 'left', 'right']].values
    #And the steering command in a variable y (output data)
    y = data_df['steering'].values

    # With the help of the SciKit train_test_split function, we can
    # easily split the data into training data and validation data
    # The amount of the total data that is used for validation is controlled
    # by the parameter 'test_size'.
    # So, a test size of 0.2 would use 80% of data for training and 20%
    # for validation

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
```

Then, we build the neural network. The choice of layers was made following this [Nvidia paper](https://arxiv.org/pdf/1604.07316.pdf), describing a solution found by the Nvidia researchers. We kept the same architecture neural network architecture since it is very optimized and could save us weeks of trial and error in order to find a viable solution.

```python
# For the construction of this neural network, the models library from Keras
# was used since it simplifies the process

# First, we tell Keras that we want a sequential model = a linear stack of layers

model = Sequential()

# Then we create a layer that will normalize the images (allows to avoid saturation
# and makes the gradients work better)

model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))

# Then, we add 5 convolution layers. The idea behind these layers is to
# "kernelize" the image. It will separate an image in a multiple set of
# smaller images, in order to facilitate feature recognition.
# for the parameters of the layers we took the one described in the paper.

model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))

# The dropout layer helps prevent the overfitting by removing the useless nodes

model.add(Dropout(args.keep_prob))

# flattens the data before entering the dense part

model.add(Flatten())

# These layers are the layers that will choose the steering angle following
# the data that the convolution layers created. we can see that we start
# from 100 neurons to finish with one, the output. Again, the sequence was
# taken from the Nvidia Paper

model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()
```

Finally, we have the training part of the model. In this part we will train the NN multiple times and keep the best model of it.

The training is done in three steps:

* Define the 'checkpoints' through the "ModelCheckpoint" Keras function. This allows to tell the training program how the model should be saved. Here we use a .h5 file type output and that we save only when the epoch is better than the last better one (when the error is minimized).

* Compile the model and define how we want to define the error and the optimizer that will influence the learning factor (here we used Adam, following the [||Source||](https://github.com/llSourcell/How_to_simulate_a_self_driving_car) code).
For the error, we chose the mean_squared_error that works this way:

    * square the difference between supposed value and the value we got
    * add up all those differences for as many data points as we have
    * divide by the total number
This gives us the mean squared error that we want to minimize through gradient descent

```python
model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))
```

* Finally, we train the compiled model with the data generated through a generator (that will be described later)

```python
model.fit_generator(batch_generator(args.data_dir, X_train, y_train,    args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)
```

This function takes for arguments:

* the parameters we set at the startup (number of epochs and sample per epochs)
* the data batch generator (for the training as well as for the validation)
* the checkpoint parameters we defined earlier
* the maximal size of the queue for the generator

At the end of all this process we get a set of models that can be fed in the drive.py program (one at the time). Those models are neural networks with fixed weights that will process the images they are fed with, outputting a steering angle.

### The utils and the generator : utils.py
The generator mentioned above is simply a function that works in the following steps:

* Get the datasets split from the train_test_split mentioned above
* Chose one image from the set and its corresponding steering angle
* In order to improve the dataset, some randomly chose images will be transformed (changing brightness, flipping it,...)
* Apply a pre-processing :

    * remove the trunk and the sky from the image
    * Blur the street details a bit
    * Change the colors from RGB to YUV (works better with edge detection)
    * Apply a canny filter (Shows only the edges)
    * Bring it back to RGB (because the generator takes a 3 layers image input while canny has only 2)
    * Return the image with only the edges
* Return two arrays containing the images pre-processed and their corresponding steering angle

### Driving the car : drive.py
In this program, the code uses the model generated before to make decisions for the steering angle in real time. The part of the code doing this is the following:
```python

# Load the image sent by the simulator into an array
image = np.asarray(image)
# Use the same processing on the image than we used on the training images
image = utils.preprocess(image)
# put it un a 4D array (model requirement)
image = np.array([image])
# Use the model created earlier to predict the steering angle
steering_angle = float(model.predict(image, batch_size=1))
#send the steering angle instruction to the simulator
send_control(steering_angle, throttle)
```
The rest of the code is mainly composed by the communication methods with the simulator and will not be described in this report.



## Contributor

* Puissant Baeyens Victor, 12098, [MisterTarock](https://github.com/MisterTarock)
* De Keyzer  Paolo, 13201, [TouchTheFishy](https://github.com/TouchTheFishy)

## References

* Our "Artificial Intelligence" course
* The self-driving simulator from Udacity : https://github.com/udacity/self-driving-car-sim
* The self-driving car project from Mr. S. Raval. : https://github.com/llSourcell/How_to_simulate_a_self_driving_car
* The Nvidia paper "End to End Learning for Self-Driving Cars" : https://arxiv.org/pdf/1604.07316.pdf
