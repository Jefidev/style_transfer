import os

import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image


class ImageLoader(object):

    def __init__(self, file_path):
        print(os.getcwd())
        self.img = Image.open(file_path)

        w, h = self.img.size
        self.width = w
        self.height = h

    def resize(self, width, height):

        self.img = self.img.resize((width, height))

        w, h = self.img.size
        self.width = w
        self.height = h

    def get_as_array(self):
        return np.expand_dims(img_to_array(self.img), axis=0)
