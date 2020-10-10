#!/usr/bin/env python3
"""Neural Style Tranfer"""

import numpy as np
import tensorflow as tf


class NST:
    """performs tasks for neural style transfer
    Class atributes:
        - Content layer where will pull our feature maps
        - Style layer we are interested in
    """

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """ initializing the varibles
        Arg:
            - style_image: img used as a style reference, numpy.ndarray
            - content_image: image used as a content reference, numpy.ndarray
            - alpha: the weight for content cost
            - beta: the weight for style cost
        Enviornment:
            Eager execution: TensorFlow’s imperative programming
                             environment, evaluates operations immediately
        """
        c_error = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(style_image, np.ndarray):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if len(style_image.shape) != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray):
            raise TypeError(c_error)
        if len(content_image.shape) != 3:
            raise TypeError(c_error)
        if content_image.shape[2] != 3:
            raise TypeError(c_error)

        if isinstance(alpha, str):
            raise TypeError("alpha must be a non-negative number")
        if alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if isinstance(beta, str):
            raise TypeError("beta must be a non-negative number")
        if beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """ rescales an image such that its pixels values are between 0
            and 1 and its largest side is 512 pixels
        Arg:
           - image: np.ndarray (h, w, 3) containing the image to be scaled
        Returns:
           - A scaled image Tensor
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        if len(image.shape) != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        if image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        image = np.expand_dims(image, axis=0)
        bicubic = tf.image.ResizeMethod.BICUBIC
        image = tf.image.resize_images(image,
                                       (512, 512),
                                       method=bicubic,
                                       preserve_aspect_ratio=True)
        image = tf.cast(image, tf.float32)
        image = image / 255
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)

        return image

    def load_model(self):
        """ load the model VGG19 used to calculate the cost
            and access to the intermediate layers.
        Saved:
            keras model that takes image inputs and outputs the style and
            content intermediate layers.
        Return: void function
        """
        vgg_load = tf.keras.applications.vgg19.VGG19(include_top=False,
                                                     weights='imagenet')

        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg_load.save("base_model")

        vgg = tf.keras.models.load_model("base_model",
                                         custom_objects=custom_objects)

        style_outputs, content_outputs = [], []
        # Get output layers corresponding to style and content layers
        for layer in vgg.layers:
            layer.trainable = False
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)
            if layer.name in self.content_layer:
                content_outputs.append(layer.output)

        model_outputs = style_outputs + content_outputs
        # Build model

        self.model = tf.keras.models.Model(vgg.input, model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """ calculate the gram matrices
        Arg:
         - input_layer: instance of tf.Tensor or tf.Variable of shape
                    (1, h, w, c) with layer.output to calculate gram matrix
        Returns: tf.Tensor shape (1, c, c) with gram matrix of input_layer
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if (len(input_layer.shape)) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        channels = int(input_layer.shape[-1])
        activation = tf.reshape(input_layer, [-1, channels])
        n = tf.shape(activation)[0]
        gram = tf.matmul(activation, activation, transpose_a=True)
        gram_matrix = gram / tf.cast(n, tf.float32)
        gram_matrix = tf.expand_dims(gram_matrix, axis=0)

        return gram_matrix

    def generate_features(self):
        """extract the style and content features
        Arg:
           - gram_style_features: gram matrices list calculated from the
                                  style layer outputs of the style image
           - content_feature: content layer output of the content image
        Returns:
           returns the style features and the content features.
        """
        nl_style = len(self.style_layers)
        # load and process the images: content and style image
        content_img = self.content_image
        h = int(content_img.shape[1])
        w = int(content_img.shape[2])
        style_img = self.style_image
        style_img = tf.image.resize_image_with_crop_or_pad(style_img, h, w)
        # batch compute content and style features
        vgg19 = tf.keras.applications.vgg19
        content = vgg19.preprocess_input(content_img * 255)
        style = vgg19.preprocess_input(style_img * 255)
        # Get the style and content feature representations from our model
        out_content = self.model(content)
        outs_style = self.model(style)

        self.gram_style_features = [self.gram_matrix(
            style_layer
        ) for style_layer in outs_style[:nl_style]]

        self.content_feature = out_content[-1]

    def layer_style_cost(self, style_output, gram_target):
        """ calculate the style cost for a single layer
        Arg:
           - style_output: tf.Tensor (1, h, w, c) with layer style output
                           of the generated image
           - gram_target - tf.Tensor (1, c, c) the gram matrix of the target
                           style output for that layer
        Return: the layer’s style cost
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError("style_output must be a tensor of rank 4")
        if (len(style_output.shape)) != 4:
            raise TypeError("style_output must be a tensor of rank 4")
        c = (style_output.shape[3])
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))
        if (len(gram_target.shape)) != 3:
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))
        if (gram_target.shape[0] != 1):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))
        if (gram_target.shape[1] != c) or (gram_target.shape[2] != c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """ style cost for generated image
        Arg:
          - style_outputs: tf.Tensor list style outputs for the generated img
        Return: the style cost
        """
        len_style_l = (len(self.style_layers))
        if not isinstance(style_outputs, list):
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    len_style_l))
        if (len(style_outputs)) != len_style_l:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    len_style_l))

        nl_style = len(self.style_layers)
        weight_style_cost = 1.0 / float(nl_style)
        cost = 0.0
        for i, output in enumerate(style_outputs):
            cost = cost + (
                self.layer_style_cost(
                    output,
                    self.gram_style_features[i]) * weight_style_cost)

        return cost

    def content_cost(self, content_output):
        """ Calculates the content cost for the generated image
        Arg:
           - content_output: tf.Tensor with the content out for generated img
        Return:
          - The content cost
        """
        s = self.content_feature.shape
        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(s))
        if (content_output.shape != s):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(s))

        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """ calculates the cost for the generated image
        Arg:
          - generated_image: tf.Tensor shape (1, nh, nw, 3) containing the
                             generated image
        Return: (J, J_content, J_style)
            - J is the total cost
            - J_content is the content cost
            - J_style is the style cost
        """
        s = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s))
        if (generated_image.shape != s):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s))

        vgg19 = tf.keras.applications.vgg19
        preprocecced = vgg19.preprocess_input(generated_image * 255)
        outputs = self.model(preprocecced)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J = (self.alpha * J_content) + (self.beta * J_style)

        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """ calculates the gradients for the tf.Tensor
        Arg:
           - generated image of shape (1, nh, nw, 3)
        Return: (gradients, J_total, J_content, J_style)
           - gradients tf.Tensor with the gradients for the generated image
           - J_total is the total cost for the generated image
           - J_content is the content cost for the generated image
           - J_style is the style cost for the generated image
        """
        s = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s))
        if (generated_image.shape != s):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s))

        with tf.GradientTape() as tape:
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style
