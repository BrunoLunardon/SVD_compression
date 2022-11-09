import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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

# for k in [0.01, 0.05, 0.1, 0.15, 0.25]:
#     for entry in ['img_for_compression/mulher_colorida.jpg', 'img_for_compression/senhor.jpg', 'img_for_compression/galaxia.jpg', "img_for_compression/new_york.jpg"]:
#         red_channel, green_channel, blue_channel = split_image(open_image(entry))
#         r=compress_channel(red_channel, k)
#         g=compress_channel(green_channel, k)
#         b=compress_channel(blue_channel, k)
#         combine_image(r,g,b).save("compressed_images/_"+str(k)+entry[19:])


# U,S,V=np.linalg.svd(open_image_bw('img_for_compression/rio.jpg'),full_matrices=False)
# S=np.diag(S)

# plt.figure(1)
# plt.title("Valores Singulares")
# plt.semilogy(np.diag(S))
# plt.savefig(f'Plots/singular_values_rio.jpg')

# plt.figure(2)
# plt.title('Soma cumulativa dos valores singulares')
# plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
# plt.savefig(f'Plots/rio_cumulative.jpg')


# U,S,V=np.linalg.svd(open_image_bw('img_for_compression/new_york.jpg'),full_matrices=False)
# S=np.diag(S)

# plt.figure(3)
# plt.title("Valores Singulares")
# plt.semilogy(np.diag(S))
# plt.savefig(f'Plots/singular_values_york.jpg')

# plt.figure(4)
# plt.title('Soma cumulativa dos valores singulares')
# plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
# plt.savefig(f'Plots/york_cumulative.jpg')

# for k in [0.05,0.1,0.15,0.25]:
#     with os.scandir('img_for_compression') as entries:
#         for entry in entries:
#             if entry.is_file():
#                 red_channel, green_channel, blue_channel = split_image(open_image(entry.path))
#                 r=compress_channel(red_channel, k)
#                 g=compress_channel(green_channel, k)
#                 b=compress_channel(blue_channel, k)
#                 combine_image(r,g,b).save(f"compressed_images/_{k}_"+entry.name)

# with os.scandir('img_for_compression') as entries:
#     for entry in entries:
#         if entry.is_file():
#             img=open_image(entry.path)
#             img_matrix=np.array(img)
#             U, S, V=np.linalg.svd(img_matrix,full_matrices=False)
#             S=np.diag(S)

#             plt.figure()
#             plt.title("Valores Singulares")
#             plt.semilogy(np.diag(S))
#             plt.savefig(f'Plots/singular_values_{entry.name}')

#             plt.figure()
#             plt.title('Soma cumulativa dos valores singulares')
#             plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
#             plt.savefig(f'Plots/cumulative_{entry.name}')




# red_channel, green_channel, blue_channel = split_image(open_image('img_for_compression/bike.jpg'))
# r=compress_channel(red_channel, 0.2)
# g=compress_channel(green_channel, 0.2)
# b=compress_channel(blue_channel, 0.2)
# combine_image(r,g,b).save(f"compressed_images/bike_0.2rgb.jpg")


# img=open_image_bw('img_for_compression/bike.jpg')
# compressed=compress_channel(img, 0.5)

U, S, V=np.linalg.svd(open_image_bw('img_for_compression/bike.jpg'),full_matrices=False)
S=np.diag(S)

plt.figure()
plt.title("Valores Singulares")
plt.semilogy(np.diag(S))
plt.savefig(f'Plots/singular_values_bike.jpg')

plt.figure()
plt.title('Soma cumulativa dos valores singulares')
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.savefig(f'Plots/bike_cumulative.jpg')
