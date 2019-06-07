import cv2, os
import numpy as np
import matplotlib.image as mpimg


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    image= cv2.imread(os.path.join(data_dir, image_file.strip()))
    
    return image


def preprocess(image):
    
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    image = cv2.bilateralFilter(image,10,50,75)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
    image=cv2.Canny(image,10,20)
    #cv2.imshow("test",image)
    #cv2.waitKey()
    image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    
    
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]

            image = load_image(data_dir, center)
            # add the image and steering angle to the batch
            image=preprocess(image)
            images[i] =image
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers
