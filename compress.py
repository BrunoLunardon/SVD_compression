import numpy as np
from PIL import Image
import cv2


def open_image(image_path):
    """_summary_: Open the image

    :param image_path: path to the image
    :type image_path: str
    :return: image
    :rtype: PIL.Image.Image
    """

    image = Image.open(image_path)
    return image


def split_image(image):
    """_summary_: Split the image into three channels

    :param image: Image object
    :type image: PIL.Image.Image
    :return: Three channels of the image
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """

    image_matrix = np.array(image)
    red_channel = image_matrix[:, :, 0]
    green_channel = image_matrix[:, :, 1]
    blue_channel = image_matrix[:, :, 2]
    return red_channel, green_channel, blue_channel

def compress_channel(channel, k):
    """_summary_: Compress the image by j-means clustering
    
    :param channel: the channel of the image
    :type channel: numpy.ndarray
    :param k: the percentage of clusters relating to the smallest dimension of the image
    :type k: float ranging from 0 to 1
    :return: the compressed image
    :rtype: numpy.ndarray
    """

    j=int(k*min(channel.shape))
    U, S, V = np.linalg.svd(channel)
    compressed_channel = np.dot(U[:, :j], np.dot(np.diag(S[:j]), V[:j, :]))
    return compressed_channel
    
red_channel, green_channel, blue_channel = split_image(open_image('dogs.jpg'))

r=compress_channel(red_channel, 0.1)
g=compress_channel(green_channel, 0.1)
b=compress_channel(blue_channel, 0.1)


def combine_image(red_channel, green_channel, blue_channel):
    """_summary_: Combine the three channels into one image

    :param red_channel: red channel of the image
    :type red_channel: numpy.ndarray
    :param green_channel: green channel of the image
    :type green_channel: numpy.ndarray
    :param blue_channel: blue channel of the image
    :type blue_channel: numpy.ndarray
    :return: combined image
    :rtype: PIL.Image.Image
    """

    image_matrix = np.dstack((red_channel, green_channel, blue_channel))
    img=image_matrix.astype(np.uint8)
    img_2=Image.fromarray(img)
    return img_2

combine_image(r,g,b).show()




