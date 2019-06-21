import cv2, os
import numpy as np
import matplotlib.image as mpimg

#parameters used for the resizing
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

#Imports the images and puts them in a matrix
def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def preprocess(image):
    image = image[60:-25, :, :] # remove the sky and the car front
    #remove some of the sides
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    #blur the details of the road
    image = cv2.bilateralFilter(image,7,50,75)
    #change to YUV since it works better to apply canny
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    #canny to exctact edges
    image = cv2.Canny (image,5,20)
    #back to a 3 layer image since the convo need 3 layers as input
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

#the following randomizing steps are reused from the ||source|| repo
def choose_image(data_dir, center, left, right, steering_angle):
    
    #Randomly choose an image from the center, left or right, and adjust
    #the steering angle.
    
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle

    
def random_flip(image, steering_angle):
    #flip the image and adapt the steering angle
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    #move the image and adjust the steering angles proportionnally
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    #changes the brightness of the image
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    #apply the forementioned transformations on the image
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle

#create a batch of training (or validation) images 
def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    #create two arrays that will contain the generated set of paths of the images and the steering angles
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        #take a random image in the recorded batch
        for index in np.random.permutation(image_paths.shape[0]):
            # save the various paths of the various angle of the image in 3 variables
            center, left, right = image_paths[index]
            #same for the steering angle
            steering_angle = steering_angles[index]
            # randomly chose if the image will be transformed
            if is_training and np.random.rand() < 0.6:
                #if yes, use the augment function
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                #if not, just load the image
                image = load_image(data_dir, center) 
            # add the images and the steering angle to the array
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            #end when the loop has been achieved the desired amount of times (defined in the parameters at the startup of the training)
            if i == batch_size:
                break
        yield images, steers