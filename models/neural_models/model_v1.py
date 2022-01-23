#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

from tensorflow.keras import Input
from tensorflow.keras import activations
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from models.neural_models.neural_model import NeuralModel
import numpy


class ModelsV1(NeuralModel):

    def __init__(self, args):

        super().__init__(args)
        self.create_neural_network()

    def create_neural_network(self):

        input_layer_block = Input(shape=(self.feature_window_width, self.feature_window_length, 1))

        first_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(input_layer_block)
        first_convolution = Activation(activations.relu)(first_convolution)

        second_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(first_convolution)
        second_convolution = Activation(activations.relu)(second_convolution)

        third_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(second_convolution)
        third_convolution = Activation(activations.relu)(third_convolution)

        fourth_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(third_convolution)
        fourth_convolution = Activation(activations.relu)(fourth_convolution)

        fifth_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(fourth_convolution)
        fifth_convolution = Activation(activations.relu)(fifth_convolution)

        sixth_convolution = Conv2D(180, (3, 3), strides=(2, 2), padding='same')(fifth_convolution)
        sixth_convolution = Activation(activations.relu)(sixth_convolution)

        first_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(sixth_convolution)
        first_deconvolution = Activation(activations.relu)(first_deconvolution)

        interpolation = Add()([first_deconvolution, fifth_convolution])

        second_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        second_deconvolution = Activation(activations.relu)(second_deconvolution)

        interpolation = Add()([second_deconvolution, fourth_convolution])

        third_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        third_deconvolution = Activation(activations.relu)(third_deconvolution)

        interpolation = Add()([third_deconvolution, third_convolution])

        fourth_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        fourth_deconvolution = Activation(activations.relu)(fourth_deconvolution)

        interpolation = Add()([fourth_deconvolution, second_convolution])

        fifth_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        fifth_deconvolution = Activation(activations.relu)(fifth_deconvolution)

        interpolation = Add()([fifth_deconvolution, first_convolution])

        sixth_deconvolution = Conv2DTranspose(180, (3, 3), strides=(2, 2), padding='same')(interpolation)
        sixth_deconvolution = Activation(activations.relu)(sixth_deconvolution)

        interpolation = Add()([sixth_deconvolution, input_layer_block])

        convolution_model = Conv2DTranspose(180, (3, 3), strides=(1, 1), padding='same')(interpolation)
        convolution_model = Activation(activations.relu)(convolution_model)

        convolution_model = Conv2DTranspose(180, (3, 3), strides=(1, 1), padding='same')(convolution_model)
        convolution_model = Activation(activations.relu)(convolution_model)

        convolution_model = Conv2D(1, (1, 1))(convolution_model)

        convolution_model = Model(input_layer_block, convolution_model)
        convolution_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.model = convolution_model

    @staticmethod
    def check_feature_empty(list_feature_samples):

        number_true_samples = 0

        for i in list_feature_samples:

            for j in i:

                if int(j) == 1:
                    number_true_samples += 1

        if number_true_samples > 0:
            return 1

        return 0

    def remove_empty_features(self, x_training, y_training):

        x_training_list = []
        y_training_list = []

        for i in range(len(x_training)):

            if self.check_feature_empty(x_training[i]):
                x_training_list.append(x_training[i])
                y_training_list.append(y_training[i])

        return numpy.array(x_training_list), numpy.array(y_training_list)

    def training(self, x_training, y_training, evaluation_set):

        x_training, y_training = self.remove_empty_features(x_training, y_training)

        for i in range(self.epochs):

            random_array_feature = self.get_random_batch(x_training)
            batch_training_in = self.get_feature_batch(x_training, random_array_feature)
            batch_training_out = self.get_feature_batch(y_training, random_array_feature)
            self.model.fit(x=batch_training_in, y=batch_training_out, epochs=1, verbose=1)

            if i % 10 == 0:
                feature_predicted = self.model.predict(batch_training_in[0:10])
                self.save_image_feature(feature_predicted[0], batch_training_out[0], batch_training_in[0], i)

        return 0

discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
discriminator.summary()


latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(8 * 8 * 128),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ],
    name="generator",
)
generator.summary()

"""
## Override `train_step`
"""


class GAN(keras.Model):

    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):

        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(random_latent_vectors)
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:

            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result(),}


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("generated_img_%03d_%d.png" % (epoch, i))



epochs = 1  # In practice, use ~100 epochs

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

gan.fit(
    dataset, epochs=epochs, callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)]
)