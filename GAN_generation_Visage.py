# -*- coding: utf-8 -*-

import math
import os
import hashlib
from urllib.request import urlretrieve
import zipfile
import gzip
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt

import tool
#barkaoui

tf.reset_default_graph()

#Téléchargement du dataset celebA  
dataset_path_celebA = 'input'
tool.download_extract('celeba', dataset_path_celebA)


"""Explorer les données"""

#nombre d'images à afficher
num_images_to_show = 25

celebA_images = tool.get_batch(glob(os.path.join(dataset_path_celebA, 'img_align_celeba/*.jpg'))[:num_images_to_show], 28,
                                28, 'RGB')

plt.imshow(tool.images_square_grid(celebA_images, 'RGB'))


"""Entrées du modèle"""

# définition des entrees(inputs) du modele
def inpts_mdl(img_width, img_height, img_channels, latent_space_z_dim):
    true_inpts_mdl = tf.placeholder(tf.float32, (None, img_width, img_height, img_channels),
                                 'true_inpts_mdl')
    z_inputs = tf.placeholder(tf.float32, (None, latent_space_z_dim), 'z_inputs')
    tx_apprentissage_mdl = tf.placeholder(tf.float32, name='tx_apprentissage_mdl')
    
    return true_inpts_mdl, z_inputs, tx_apprentissage_mdl


"""Definition du Discriminateur"""
# Définition de la fonction discriminante
def discriminator(input_imgs, reuse=False):
    
    with tf.variable_scope('discriminator', reuse=reuse):
        
        leaky_param_alpha = 0.2
        # définir les couches
        layer_convult_1 = tf.layers.conv2d(input_imgs, 64, 5, 2, 'same')
        relu_outpt2 = tf.maximum(leaky_param_alpha * layer_convult_1, layer_convult_1)
        
        layer_convult_2 = tf.layers.conv2d(relu_outpt2, 128, 5, 2, 'same')
        normalized_output = tf.layers.batch_normalization(layer_convult_2, training=True)
        relu_outpt = tf.maximum(leaky_param_alpha * normalized_output, normalized_output)
        
        layer_convult_3 = tf.layers.conv2d(relu_outpt, 256, 5, 2, 'same')
        normalized_output = tf.layers.batch_normalization(layer_convult_3, training=True)
        relu_outpt2 = tf.maximum(leaky_param_alpha * normalized_output, normalized_output)
        
        # remodeler la sortie pour les logits pour être tenseur 2D
        flat_output = tf.reshape(relu_outpt2, (-1, 4 * 4 * 256))
        lay_logits = tf.layers.dense(flat_output, 1)
        output = tf.sigmoid(lay_logits)
    return output, lay_logits
        


"""Definition du Generateur"""
def generator(z_latent_space, output_channel_dim, is_train=True):
    
    with tf.variable_scope('generator', reuse=not is_train):
        # paramètre relu qui fuit
        leaky_param_alpha = 0.2
        ## Première couche entièrement connectée
        fully_connected_layer = tf.layers.dense(z_latent_space, 2*2*512)
        
        reshaped_output = tf.reshape(fully_connected_layer, (-1, 2, 2, 512))
        normalized_output = tf.layers.batch_normalization(reshaped_output, training=is_train)
        relu_outpt2 = tf.maximum(leaky_param_alpha * normalized_output, normalized_output)
        
        layer_convult_1 = tf.layers.conv2d_transpose(relu_outpt2, 256, 5, 2, 'valid')
        normalized_output = tf.layers.batch_normalization(layer_convult_1, training=is_train)
        relu_outpt2 = tf.maximum(leaky_param_alpha * normalized_output, normalized_output)
        
        layer_convult_2 = tf.layers.conv2d_transpose(relu_outpt2, 128, 5, 2, 'same')
        normalized_output = tf.layers.batch_normalization(layer_convult_2, training=is_train)
        relu_outpt2 = tf.maximum(leaky_param_alpha * normalized_output, normalized_output)
        
        lay_logits = tf.layers.conv2d_transpose(relu_outpt2, output_channel_dim, 5, 2, 'same')
        output = tf.tanh(lay_logits)
        
        return output


"""Pertes du modèle"""


## Définir l'erreur pour le discriminateur et le générateur
def loss_mdl(input_actual, input_latent_z, out_channel_dim):
    mdl_generator = generator(input_latent_z, out_channel_dim)
    discrimator_mdl_real, discrimator_logits_real = discriminator(input_actual)
    discrimator_mdl_fake, discriminator_logits_fake = discriminator(mdl_generator, reuse=True)
    
    discriminator_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=discrimator_logits_real, labels=tf.ones_like(discrimator_mdl_real)))

    discriminator_loss_fakeloss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=discriminator_logits_fake, labels=tf.zeros_like(discrimator_mdl_fake)))
    
    loss_generator = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=discriminator_logits_fake, labels=tf.ones_like(discrimator_mdl_fake)))
    
    discriminator_loss_fakeloss = discriminator_loss_real + discriminator_loss_fakeloss_fake
    
    return discriminator_loss_fakeloss, loss_generator
    
    
"""Optimiseur de modèle"""


# spécifiant les critères d'optimisation
def optimizer_mdl(discriminator_loss_fakeloss, loss_generator, learning_rate, beta1):
    trainable_vars = tf.trainable_variables()
    
    discriminator_loss_fakevars = [var for var in trainable_vars if var.name.startswith('discriminator')]
    gen_vars = [var for var in trainable_vars if var.name.startswith('generator')]
    
    discriminator_loss_faketrain_opt = tf.train.AdamOptimizer(
        learning_rate, beta1=beta1).minimize(discriminator_loss_fakeloss, var_list=discriminator_loss_fakevars)
    
    update_operations = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    gen_updates = [opt for opt in update_operations if opt.name.startswith('generator')]
    
    with tf.control_dependencies(gen_updates):
        gen_train_opt = tf.train.AdamOptimizer(
            learning_rate, beta1).minimize(loss_generator, var_list=gen_vars)
        
    return discriminator_loss_faketrain_opt, gen_train_opt
    




""" former le modele"""

def display_gen_output(sess, num_images, input_latent_z, output_channel_dim, img_mode):
    
    cmap = None if img_mode == 'RGB' else 'gray'
    latent_space_z_dim = input_latent_z.get_shape().as_list()[-1]
    examples_z = np.random.uniform(-1, 1, size=[num_images, latent_space_z_dim])

    examples = sess.run(
        generator(input_latent_z, output_channel_dim, False),
        feed_dict={input_latent_z: examples_z})

    images_grid = tool.images_square_grid(examples, img_mode)
    plt.imshow(images_grid, cmap=cmap)
    plt.show()
    

def mdl_trainning(num_epocs, train_batch_size, z_dim, learning_rate, beta1, get_batches, input_data_shape, data_img_mode):
    _, image_width, image_height, image_channels = input_data_shape

    actual_input, z_input, leaningRate = inpts_mdl(
        image_width, image_height, image_channels, z_dim)

    discriminator_loss_fakeloss, loss_generator = loss_mdl(actual_input, z_input, image_channels)

    discriminator_loss_fakeopt, gen_opt = optimizer_mdl(discriminator_loss_fakeloss, loss_generator, learning_rate, beta1)

    steps = 0
    print_every = 50
    show_every = 100
    model_loss = []
    num_images = 2
    
    with tf.Session() as sess:

        # initialiser toutes les variables
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(num_epocs):
            for batch_images in get_batches(train_batch_size):

                steps += 1
                batch_images *= 2.0
                z_sample = np.random.uniform(-1, 1, (train_batch_size, z_dim))

                _ = sess.run(discriminator_loss_fakeopt, feed_dict={
                    actual_input: batch_images, z_input: z_sample, leaningRate: learning_rate})
                _ = sess.run(gen_opt, feed_dict={
                    z_input: z_sample, leaningRate: learning_rate})

                if steps % print_every == 0:
                    train_loss_disc = discriminator_loss_fakeloss.eval({z_input: z_sample, actual_input: batch_images})
                    train_loss_gen = loss_generator.eval({z_input: z_sample})

                    print("Epoch {}/{}...".format(epoch_i + 1, num_epocs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_disc),
                          "Generator Loss: {:.4f}".format(train_loss_gen))
                    model_loss.append((train_loss_disc, train_loss_gen))

                if steps % show_every == 0:
                    display_gen_output(sess, num_images, z_input, image_channels, data_img_mode)


# Formation du modèle sur le jeu des données CelebA

train_batch_size = 64
z_dim = 100
learning_rate = 0.002
beta1 = 0.5

num_epochs = 1

"""
Exécutez vos GAN sur CelebA. Il faudra environ 45 minutes sur le GPU moyen pour exécuter une époque. 
Vous pouvez exécuter toute l'époque ou vous arrêter lorsqu'elle commence à générer des visages réalistes.
"""

celeba_dataset = tool.Dataset('celeba', glob(os.path.join(dataset_path_celebA, 'img_align_celeba/*.jpg')))

with tf.Graph().as_default():
    mdl_trainning(num_epochs, train_batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
                celeba_dataset.shape, celeba_dataset.image_mode)












