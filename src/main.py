import argparse
import time

from keras.applications import vgg16
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b

from loaders.image_loader import ImageLoader
from model.model import StyleTransferModel
from model.model_evaluater import Evaluator


def main_process():
    height = 480
    width = 480

    source = ImageLoader("data/oim.jpg")
    source.resize(width, height)

    style = ImageLoader("data/magritte.jpg")
    style.resize(source.width, source.height)

    # Building the model
    model = StyleTransferModel(source, style, source.height, source.width)
    evaluator = Evaluator(model)

    x = source.get_as_array()
    x = vgg16.preprocess_input(source.get_as_array()).flatten()

    nbr_it = 2

    for i in range(nbr_it):

        print("------------- {} --------------".format(i))
        x, min_val, info = fmin_l_bfgs_b(
            evaluator.loss, x, fprime=evaluator.grads, maxfun=20)

        generated_img = x.copy().reshape((height, width, 3))

        print(generated_img)


if __name__ == "__main__":
    main_process()
