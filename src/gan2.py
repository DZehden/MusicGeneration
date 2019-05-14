#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers
import time
from src.music_dataset import MusicDataset
from src.midi_utils import ndarray_to_midi

BATCH_SIZE = 5

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

noise_dim = 100
num_examples_to_generate = 1
seed = tf.random.normal([num_examples_to_generate, noise_dim])

class MusicGAN:
    def __init__(self, checkpoint_dir, load_model=False):
        """
        Creates a MusicGAN Object

        :param checkpoint_dir: str
            filepath to checkpoint directory
        :param load_model: boolean
            flag indicating whether or not to load model from checkpoint
        """
        gen = MusicGAN.make_generator_model()
        disc = MusicGAN.make_discriminator_model()
        gen_opt = tf.keras.optimizers.Adam(1e-4)
        disc_opt = tf.keras.optimizers.Adam(1e-4)
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                              discriminator_optimizer=disc_opt,
                                              generator=gen,
                                              discriminator=disc)
        self.chkpt_man = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir, max_to_keep=3)
        if load_model:
            self.checkpoint.restore(self.chkpt_man.latest_checkpoint)
        self.generator = gen
        self.discriminator = disc
        self.generator_optimizer = gen_opt
        self.discriminator_optimizer = disc_opt

    @staticmethod
    def make_generator_model():
        """
        Creates a generator CNN for GAN

        :return: tf.keras.Sequential
        """
        model = tf.keras.Sequential()
        model.add(layers.Dense(6*8*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((6, 8, 256)))
        print('1')
        print(model.output_shape)
        assert model.output_shape == (None, 6, 8, 256) # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        print('2')
        print(model.output_shape)
        assert model.output_shape == (None, 6, 8, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (4, 4), strides=(4, 4), padding='same', use_bias=False))
        print('3')
        print(model.output_shape)
        assert model.output_shape == (None, 24, 32, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(32, (4, 2), strides=(4, 2), padding='same', use_bias=False))
        print('4')
        print(model.output_shape)
        assert model.output_shape == (None, 96, 64, 32)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (4, 2), strides=(4, 2), padding='same', use_bias=False, activation='relu'))
        print('5')
        print(model.output_shape)
        assert model.output_shape == (None, 384, 128, 1)

        return model

    @staticmethod
    def make_discriminator_model():
        """
        Create discriminator for GAN

        :return: tf.keras.Sequential
        """
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(32, (4, 2), strides=(4, 2), padding='same',
                                input_shape=[384, 128, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (4, 4), strides=(4, 4), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (4, 4), strides=(4, 4), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    @staticmethod
    def discriminator_loss(real_output, fake_output):
        """
        Loss function for discriminator model

        :param real_output: tf.Tensor
            real training data
        :param fake_output: tf.Tensor
            data generated by generator
        :return: np.float
        """
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    def generator_loss(fake_output):
        """
        Loss function for generator

        :param fake_output:
            data generated by generator
        :return: np.float
        """
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def train(self, dataset, epochs):
        """
        Trains model on a dataset for a certain number of epochs

        :param dataset: MusicDataset
             Dataset to train model on
        :param epochs: int
            number of epochs to train for
        :return: None
        """
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                resized_in = np.resize(image_batch,(1,384,128,1))
                resized_in = resized_in.astype('float32')
                train_step(resized_in, self.generator, self.generator_optimizer, self.discriminator, self.discriminator_optimizer)

            self.generate_and_save_audio(self.generator, epoch + 1, seed)

            # Save the model every 15 epochs
            if (epoch) % 15 == 0:
                self.checkpoint.save(file_prefix = checkpoint_prefix)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        self.generate_and_save_audio(self.generator, epochs, seed)

    def generate_and_save_audio(self, model, epoch, test_input):
        """
        Generate an artificial output for a test input

        :param model: tf.keras.Sequential
            Generator to make prediction
        :param epoch: int
            epoch number
        :param test_input: tf.tensor
            Random number array to seed model with
        :return: None
        """
        predictions = model(test_input, training=False)
        predictions = np.reshape(predictions,(384,128))
        ndarray_to_midi(predictions, './../output/' + str(epoch) + '.mid')

    @staticmethod
    def load_model(checkpoint_dir):
        res = tf.train.Checkpoint.restore(checkpoint_dir)
        return res

@tf.function
def train_step(images, generator, generator_optimizer, discriminator, discriminator_optimizer):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = MusicGAN.generator_loss(fake_output)
        disc_loss = MusicGAN.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

