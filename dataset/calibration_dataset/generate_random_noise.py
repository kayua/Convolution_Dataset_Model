import glob
import random

import cv2
from tqdm import tqdm

DEFAULT_PATH_ORIGINAL_IMAGES = 'original_image/'
DEFAULT_PATH_FAILED_IMAGES = 'failed_image/'
DEFAULT_THRESHOLD_NOISE = 75


def generate_noise_in_image(file_image):

    img = cv2.imread(file_image, cv2.IMREAD_GRAYSCALE)

    height, width = img.shape

    for x in range(0, height):

        for y in range(0, width):

            random_number = random.randint(0, 100)

            if random_number < DEFAULT_THRESHOLD_NOISE:

                img[x, y] = 0 #random.randint(0, 255)


    cv2.imwrite(DEFAULT_PATH_FAILED_IMAGES+'/'+str(file_image.split('/')[-1]), img)



for  i in tqdm(glob.glob(DEFAULT_PATH_ORIGINAL_IMAGES+'*')):
    generate_noise_in_image(i)
