#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers
import time
from src.music_dataset import MusicDataset
from src.midi_utils import ndarray_to_midi

dataset = MusicDataset('../MusicMats/MusicMats')

#BUFFER_SIZE should equal total number of tracks used
BATCH_SIZE = 1


def make_generator_model():
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
    
    model.add(layers.Conv2DTranspose(1, (4, 2), strides=(4, 2), padding='same', use_bias=False, activation='tanh'))
    print('5')
    print(model.output_shape)
    assert model.output_shape == (None, 384, 128, 1)

    return model



def make_discriminator_model():
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



generator = make_generator_model()
discriminator = make_discriminator_model()


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)



checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 1
seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            resized_in = np.resize(image_batch,(1,384,128,1))
            resized_in = resized_in.astype('float32')
            train_step(resized_in)
        
        generate_and_save_audio(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_audio(generator, epochs, seed)


def generate_and_save_audio(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = np.reshape(predictions,(384,128))
    ndarray_to_midi(predictions, './output/' + str(epoch) + '.mid')




train(dataset, EPOCHS)

