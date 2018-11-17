from keras import backend as K
from keras.applications import vgg16


class StyleTransferModel():

    def __init__(self, img, style, height, width):
        self.channels = 3
        self.content_weight = 0.5
        self.total_variation_weight = 1e-4

        self.height = height
        self.width = width

        img_tensor = vgg16.preprocess_input(img.get_as_array())
        style_tensor = vgg16.preprocess_input(style.get_as_array())

        self.generated_img_tensor = K.placeholder(
            (1, img.height, img.width, 3))

        # Creating model input
        input = K.concatenate(
            [img_tensor, style_tensor, self.generated_img_tensor], axis=0)

        self.model = vgg16.VGG16(input_tensor=input,
                                 weights="imagenet", include_top=False)

        self.create_loss_tensor()

    def create_loss_tensor(self):
        loss = K.variable(0.0)
        total_variation_weight = 1e-4

        style_weight = [0.1, 0.15, 0.2, 0.25, 0.3]

        model_layers = {l.name: l.output for l in self.model.layers}

        # Getting the last convolution layer to compute content loss
        conv4 = model_layers["block4_conv2"]
        target_img_features = conv4[0, :, :, :]
        combination_img_features = conv4[2, :, :, :]

        loss += self.content_weight * \
            self._content_loss(target_img_features, combination_img_features)

        # getting all the convolution layers and compute the style loss
        style_layers = ['block{}_conv2'.format(o) for o in range(1, 6)]

        for layer_name, style_w in zip(style_layers, style_weight):

            conv = model_layers[layer_name]
            style_img_features = conv[0, :, :, :]
            combination_img_features = conv[2, :, :, :]

            style_loss = self._style_loss(
                style_img_features, combination_img_features)

            loss += style_loss * style_w

        # Adding variation loss :
        loss += total_variation_weight * \
            self._total_variation_loss(self.generated_img_tensor)

        # Create the loss and gradient function
        grad = K.gradients(loss, self.generated_img_tensor)[0]
        self.get_loss_and_grads = K.function(
            [self.generated_img_tensor], [loss, grad])

    def _content_loss(self, base, combination):
        return K.sum(K.square(combination - base))

    def _style_loss(self, style, combination):

        S = self._build_gram_matrix(style)
        C = self._build_gram_matrix(combination)

        size = self.height * self.width
        s_loss = K.sum(K.square(S - C)) / \
            (4*(self.channels ** 2) * (size ** 2))

        return s_loss

    def _build_gram_matrix(self, x):
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram_matrix = K.dot(features, K.transpose(features))

        return gram_matrix

    def _total_variation_loss(self, x):
        a = K.square(
            x[:, :self.height - 1, :self.width - 1, :] - x[:, 1:, :self.width - 1, :])
        b = K.square(
            x[:, :self.height - 1, :self.width - 1, :] - x[:, :self.height - 1, 1:, :])

        return K.sum(K.pow(a + b, 1.25))
