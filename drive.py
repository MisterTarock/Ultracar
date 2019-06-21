import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

#load our saved model
from keras.models import load_model

#Our preprocessing class - It generate also the different batches of data
import utils

# create a Socket.IO server for the communication between drive.py and the simulator
sio = socketio.Server()
#init our model and image array as empty
model = None
prev_image_array = None

#set min/max speed for our autonomous car
#speed define above 30 to be sure to drive at full speed
MAX_SPEED = 35
MIN_SPEED = 10

#and a speed limit
speed_limit = MAX_SPEED

# event sent by the simulator
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        # Use your model to compute steering and throttle
        # The values send to the simulator during the testing phase
        try:
            image = np.asarray(image)       # from PIL image to numpy array
            image = utils.preprocess(image) # apply the preprocessing
            image = np.array([image])       # the model expects 4D array

            # predict the steering angle for the image
            steering_angle = float(model.predict(image, batch_size=1))

            # The speed is "regulated" by drive.py,
            # it will lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

            # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))

        # response to the simulator with a steer angle and throttle
        send(steering_angle, throttle)
    else:
        # Edge case
        sio.emit('manual', data={}, skip_sid=True)

# event fired when simulator connect
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send(0, 0)

# to send steer angle and throttle to the simulator
def send(steering_angle, throttle):
    sio.emit(
        "steer",
         data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
            },
            skip_sid=True)


# simulator will connect to localhost:4567
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    #load model
    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap with a WSGI application
    app = socketio.WSGIApp(sio)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
