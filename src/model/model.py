from keras import backend as K
from keras.applications import vgg16


class StyleTransferModel():

    def __init__(self, img, style):
        self.channels = 3
        self.content_weight = 0.5
        self.total_variation_weight = 1e-4

        img_tensor = vgg16.preprocess_input(img.get_as_array())
        style_tensor = vgg16.preprocess_input(style.get_as_array())

        self.generated_img_tensor = K.placeholder(
            (1, img.height, img.width, 3))

        # Creating model input
        input = K.concatenate(
            [img_tensor, style_tensor, self.generated_img], axis=0)

        self.model = vgg16(input_tensor=input,
                           weights="imagenet", include_top=False)

        # Defining some loss variable

    def _content_loss(self, base, combination):
        return K.sum(K.square(combination - base))

    def _style_loss(self, style, combination, width, height):

        S = self._build_gram_matrix(style)
        C = self._build_gram_matrix(combination)

        size = height * width
        s_loss = K.sum(K.square(S - C)) / \
            (4*(self.channels ** 2) * (size ** 2))

    def _build_gram_matrix(self, x):
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram_matrix = K.dot(features, K.transpose(features))

        return gram_matrix

    def _total_variation_loss(self, x, height, width):
        a = K.square(
            x[:, :height - 1, :width - 1, :] - x[:, 1:, :width - 1, :])
        b = K.square(
            x[:, :height - 1, :width - 1, :] - x[:, :height - 1, 1:, :])
