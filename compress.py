import numpy as np
from PIL import Image
import os

def open_image(image_path):
    """_summary_: Open the image

    :param image_path: path to the image
    :type image_path: str
    :return: image
    :rtype: PIL.Image.Image
    """

    image = Image.open(image_path)
    return image

def open_image_bw(image_path):
    """_summary_: Open the image

    :param image_path: path to the image
    :type image_path: str
    :return: image_matrix
    :rtype: numpy.ndarray
    """

    image = Image.open(image_path).convert('L')
    image_matrix = np.array(image)
    return image_matrix


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
    :param k: the percentage of the singularm values used
    :type k: float range(0,1)
    :return: the compressed image
    :rtype: numpy.ndarray
    """
    
    j=int(min(channel.shape)*k)
    U, S, V = np.linalg.svd(channel, full_matrices=False)
    compressed_channel = np.dot(U[:, :j], np.dot(np.diag(S[:j]), V[:j, :]))
    return compressed_channel
    
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

def matrix_to_image(matrix):
    """_summary_: Convert the matrix into an image

    :param matrix: the matrix of the image
    :type matrix: numpy.ndarray
    :return: the image
    :rtype: PIL.Image.Image
    """

    img=matrix.astype(np.uint8)
    img = Image.fromarray(img)
    return img 

#combine_image(r,g,b).show()

with os.scandir('img_for_compression') as entries:
    for entry in entries:
        if entry.is_file():
            red_channel, green_channel, blue_channel = split_image(open_image(entry.path))
            r=compress_channel(red_channel, 100)
            g=compress_channel(green_channel, 100)
            b=compress_channel(blue_channel, 100)
            combine_image(r,g,b).save('compressed_images/compressed_'+entry.name)


#make the plots of cumulative sums
#make the plots of logs of singular values

img=compress_channel(open_image_bw('img_for_compression/dogs.png'), 0.1)
a=matrix_to_image(img)
a.save('compressed_images/compressed_dogs.png')