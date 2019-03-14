from operator import mul
from functools import reduce
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

import src.nn as nn

def get_model(im_size, num_channels, num_filters, num_encoder_layers,
              num_dense_encoder_layers, num_decoder_layers,
              num_dense_decoder_layers):
    momentum=0.9
    ema = tf.train.ExponentialMovingAverage(decay=momentum)
    store = {}
    arg_scope_layers = [nn.conv_layer, nn.deconv_layer, nn.dense_layer, nn.batch_norm]
    def encoder(x):
        counters = {}
        x_shape = nn.get_shape_as_list(x)
        with arg_scope(arg_scope_layers, counters=counters, graph_name="encoder", ema=ema, store=store):
            # encoder conv layers
            for _ in range(num_encoder_layers):
                x = nn.conv_layer(x, num_filters=num_filters, kernel_size=(3, 3), strides=(1,1))

            x = tf.reshape(x, [x_shape[0],-1])
            num_units = num_filters*im_size*im_size
            for _ in range(num_dense_encoder_layers):
                x = nn.dense_layer(x, num_units=num_units)

            h_means, h_isp_std = tf.split(x, 2, 1)
            h_std = tf.nn.softplus(h_isp_std)
        return h_means, h_std

    def decoder(h_means, h_std):
        counters = {}
        with arg_scope(arg_scope_layers, counters=counters, graph_name="decoder", ema=ema, store=store):
            h_shape = nn.get_shape_as_list(h_means)
            # Sample from distribution
            h_sample = tf.random_normal(h_shape)*h_std+h_means
            z = h_sample

            num_units = h_shape[1]
            for _ in range(num_dense_decoder_layers):
                z = nn.dense_layer(z, num_units=num_units)

            z = tf.reshape(z, [h_shape[0],im_size,im_size,-1])

            # decoder layers
            for _ in range(num_decoder_layers-1):
                z = nn.deconv_layer(z, num_filters=num_filters, kernel_size=(3,3), strides=(1,1))
            logits = nn.deconv_layer(z, num_filters=num_channels, kernel_size=(3,3), strides=(1,1), nonlinearity=tf.identity)
        return logits

    return encoder, decoder


def vae_loss(x, logits, h_means, h_std):
    h_shape = nn.get_shape_as_list(h_means)
    # KL divergence term:
    kl_loss_array = -0.5 * (1 + 2 * tf.log(h_std+1e-8) - h_means ** 2 - h_std**2)
    kl_loss = tf.reduce_sum(kl_loss_array, axis=1)
    # cross entropy term:
    cross_entropy_array = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=x,logits=logits)
    cross_entropy_loss = tf.reduce_sum(cross_entropy_array, axis=[1, 2, 3])
    nn.assert_finite(kl_loss_array)
    nn.assert_finite(cross_entropy_loss)

    return tf.reduce_mean((kl_loss + cross_entropy_loss))