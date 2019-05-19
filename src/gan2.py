#usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers
import time
from src.midi_utils import ndarray_to_midi, ndarray_to_npz
import pypianoroll
import datetime

noise_dim = 100
num_examples_to_generate = 1
seed = tf.random.normal([num_examples_to_generate, noise_dim])


class MusicGAN:
    def __init__(self, checkpoint_dir, output_dir, batch_size, load_model=False):
        """
        Creates a MusicGAN Object

        :param checkpoint_dir: str
            filepath to checkpoint directory
        :param load_model: boolean
            flag indicating whether or not to load model from checkpoint
        """

        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        gen = MusicGAN.make_generator_model()
        disc = MusicGAN.make_discriminator_model()
        gen_opt = tf.keras.optimizers.Adagrad(1e-4)
        disc_opt = tf.keras.optimizers.Adagrad(1e-6)
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                              discriminator_optimizer=disc_opt,
                                              generator=gen,
                                              discriminator=disc)
        self.model_loaded = load_model
        self.chkpt_man = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)
        if load_model:
            self.checkpoint.restore(self.chkpt_man.latest_checkpoint)
            print('Loaded checkpoint')
            print(self.chkpt_man.latest_checkpoint)
        else:
            print('No checkpoint loaded')
        self.generator = gen
        self.discriminator = disc
        self.generator_optimizer = gen_opt
        self.discriminator_optimizer = disc_opt
        self.output_dir = output_dir
        self.epochs = 0

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
        assert model.output_shape == (None, 6, 8, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 6, 8, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (4, 4), strides=(4, 4), padding='same', use_bias=False))
        assert model.output_shape == (None, 24, 32, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(32, (4, 2), strides=(4, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 96, 64, 32)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (4, 2), strides=(4, 2), padding='same', use_bias=False))
        model.add(layers.ReLU(threshold= -1, max_value=1))
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
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    @staticmethod
    def discriminator_loss(real_output, fake_output):
        """
        Loss function for discriminator model

        :param real_output: tf.Tensor
            discriminator result on real training data
        :param fake_output: tf.Tensor
            discriminator result on data created by generator
        :return: np.float
        """
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    def generator_loss(fake_output):
        """
        Loss function for generator

        :param fake_output:
            discriminator result on data generated by generator
        :return: np.float
        """
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def get_train_bools(self, fake_res, real_res, epoch):
        if (epoch < 10) and (not self.model_loaded):
            disc_train = True
            gen_train = True
            return disc_train, gen_train
        disc_train = False
        gen_train = True
        if fake_res >= 0.6 and real_res >= 0.9:
            return False, True
        if fake_res >= 0.5 and real_res >= 0.5:
            return True, True
        elif real_res >= 0.5:
            disc_train = False
            gen_train = True
            return disc_train, gen_train
        elif fake_res >= 0.5:
            disc_train = True
            gen_train = False
            return disc_train, gen_train  # Train discriminator only
        else:
            return disc_train, gen_train  # Train generator

    def train(self, dataset, epochs):
        """
        Trains model on a dataset for a certain number of epochs

        :param dataset: MusicDataset
             Dataset to train model on
        :param epochs: int
            number of epochs to train for
        :return: None
        """ 
        fake_avg = 0
        real_avg = 0
        for epoch in range(self.epochs, self.epochs + epochs):
            start = time.time()
            disc_train, gen_train = self.get_train_bools(fake_avg, real_avg, epoch)
            train_str = 'Training ' + ('discrminator, ' if disc_train else '') + ('generator' if gen_train else '')
            print(train_str)
            fake_avg = 0
            real_avg = 0
            for image_batch in dataset:
                resized_in = np.resize(image_batch, (1, 384, 128, len(dataset)))
                resized_in = resized_in.astype('float32')

                fake_res, real_res = self.train_step(resized_in, disc_train, gen_train)
                fake_avg += np.average(fake_res)
                real_avg += np.average(real_res)


                # print('Training discriminator: {0} (real_pred={1}, fake_pred, {2})'.format(disc_train, real_res, fake_res))

            # Save the model every 200 epochs
            if (epoch + 1) % 30 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                self.generate_and_save_audio(self.generator, epoch + 1, seed)
                #pred = self.predict_from_midi(self.output_dir+'50.mid')
                #print('Prediction on ' + '50.mid'+ ': {0}'.format(pred))
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            print('Average real prediction: {0}, Average fake prediction: {1}'.format(np.average(real_avg), np.average(fake_avg)))
            self.epochs += 1
            #pred = self.predict_from_midi('./output/1.mid')
            #print('Prediction: {0}'.format(pred))

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
        print('Max: {0}, min {1}, non_zero elements: {2} out of {3}'.format(predictions.max(), predictions.min(), len(predictions[predictions > 0]), 384 * 128))
        predictions = predictions * 127
        predictions = predictions.astype('uint8')
        ndarray_to_midi(predictions, self.output_dir + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")) + '_epoch_' + str(epoch) + '.mid')
        ndarray_to_npz(predictions, self.output_dir + str(datetime.datetime.now().strftime("%Y_m%_d_%H_%M")) + '_epoch_' + str(
            epoch) + '.npz')

    # TODO: Update these functions to scale data
    def predict_from_midi(self, path_to_midi):
        mtrack = pypianoroll.parse(path_to_midi)
        mat = mtrack.tracks[0].pianoroll
        resized_in = np.resize(mat, (1, 384, 128, 1))
        resized_in = resized_in.astype('float32')
        prediction = self.discriminator(resized_in, training=False)
        return prediction

    def predict_from_npz(self, path_to_npz):
        mtrack = pypianoroll.load(path_to_npz)
        mat = mtrack.tracks[0].pianoroll
        resized_in = np.resize(mat, (1, 384, 128, 1))
        resized_in = resized_in.astype('float32')
        prediction = self.discriminator(resized_in, training=False)
        return prediction

    def generate_midi(self, output_path):
        noise = tf.random.normal([1, noise_dim])
        song = self.generator(noise, training=False)[0]
        song = np.reshape(song, (384, 128))
        ndarray_to_midi(song, output_path)

    def generate_npz(self, output_path):
        noise = tf.random.normal([1, noise_dim])
        song = self.generator(noise, training=False)[0]
        song = np.reshape(song, (384, 128))
        ndarray_to_npz(song, output_path)

    @staticmethod
    def load_model(checkpoint_dir):
        res = tf.train.Checkpoint.restore(checkpoint_dir)
        return res

    def train_step(self, images, disc_train, gen_train):
        """
        Performs a single step of gradient descent on a generator and discriminator

        :param images: Dataset
            Input dataset (usually a list of or singular ndarray representing a pianoroll)
        :param generator: tf.keras.Sequential
             CNN used to generate music
        :param generator_optimizer: tf.keras.optimizers.Adam
             Function used to optimize weight of generator
        :param discriminator: tf.keras.Sequential
             CNN used to classify music as 'real' or 'fake'
        :param discriminator_optimizer: tf.keras.optimizers.Adam
             Function used to optimize weights of discriminator
        :param disc_train: bool
             Flag indicating whether or not to train the discriminator this step
        :return: None
        """
        noise = tf.random.normal([self.batch_size, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            # arr = np.reshape(generated_images, (384, 128))
            # ndarray_to_midi(arr, './output/train_sample_gen.mid')
            #
            # reroll = pypianoroll.parse('./output/train_sample_gen.mid', beat_resolution=24)
            # mat = reroll.tracks[0].pianoroll
            # generated_matrix = np.resize(mat, 1, 384, 128, 1)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            #print('Prediction on real: {}, prediction on generated: {}'.format(real_output, fake_output))
            gen_loss = MusicGAN.generator_loss(fake_output)
            disc_loss = MusicGAN.discriminator_loss(real_output, fake_output)

            if disc_train:
                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            if gen_train:
                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

        if disc_train:
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        if gen_train:
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return fake_output.numpy(), real_output.numpy()
