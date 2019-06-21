import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import argparse
import os

#for debugging, allows for reproducible (deterministic) results
np.random.seed(0)


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    #reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    #we export the images (divided per camera) in a new variable X (input data)
    X = data_df[['center', 'left', 'right']].values
    #and our steering commands as our output data
    y = data_df['steering'].values

    # With the help of the SciKit train_test_split function, we can
    # easily split the data into training data and validation data
    # The amount of the total data that is used for validation is controlled
    # by the parameter 'test_size'.
    # So, a test size of 0.2 would use 80% of data for training and 20%
    # for validation
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    NVIDIA model used
    
    """
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

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    # Define the 'checkpoints' through the "ModelCheckpoint" Keras function. 
    # This allows to tell the training program how the model should be saved. 
    # Here we use a .h5 file type output and that we save only when the epoch 
    # is better than the last better one (when the error is minimized).
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    # Compile the model and define how we want to define the error and the optimizer that will 
    # influence the learning factor (here we used Adam, following the ||Source|| code). For the error, 
    # we chose the mean_squared_error that works this way:
    # - square the difference between supposed value and the value we got
    # - add up all those differences for as many data points as we have
    # - divide by the total number This gives us the mean squared error that we want to minimize through gradient descent


    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    #Finally, we train the compiled model with the data generated through a generator (see the explanation in utils.py)

    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)

#used to convert the any input in usable data
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    data = load_data(args)
    #build model
    model = build_model(args)
    #train model on data, it saves as model.h5
    train_model(model, args, *data)


if __name__ == '__main__':
    main()