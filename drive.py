import socketio
import eventlet
import base64
from PIL import Image
from io import BytesIO

# create a Socket.IO server
sio = socketio.Server()

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
        # values send to the simulator
        steer = 0
        throttle = 1

        # response to the simulator with a steer angle and throttle
        send(steer, throttle)
    else:
        # Edge case
        sio.emit('manual', data={}, skip_sid=True)

# event fired when simulator connect
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send(0, 0)

# to send steer angle and throttle to the simulator
def send(steer, throttle):
    sio.emit("steer", data={'steering_angle': str(steer), 'throttle': str(throttle)}, skip_sid=True)


# wrap with a WSGI application
app = socketio.WSGIApp(sio)

# simulator will connect to localhost:4567
if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
