import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope


def get_name(layer_name: str, counters: dict, graph_name: str=""):
    """ utlity for keeping track of layer names """
    if counters is None:
        raise ValueError("No counter dict was provided to get_name.")
    if not layer_name in counters:
        counters[layer_name] = 0
    name = graph_name + '_' + layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name


def get_shape_as_list(x):
    """Returns the shape of the tensor as a list of ints."""
    shape = list(map(int, tf.shape(x)))
    return shape


@add_arg_scope
def conv_layer(x, num_filters, kernel_size, strides, pad="SAME", nonlinearity=None, counters=None, init=False, init_scale=1., **kwargs):
    name = get_name('conv', counters, kwargs.get("graph_name", ""))
    kernel_size=tuple(kernel_size)
    strides=tuple(strides)
    with tf.variable_scope(name):
        xshape = get_shape_as_list(x)
        V = tf.get_variable(name = "V", shape = kernel_size + (xshape[-1], num_filters), initializer=tf.random_normal_initializer(0, 0.05), dtype=tf.float32)
        b = tf.get_variable(name = "b", shape = [num_filters], initializer=tf.constant_initializer(0.), dtype=tf.float32)

        # calculate convolutional layer output
        x = tf.nn.bias_add(tf.nn.conv2d(x, V, (1,) + strides + (1,), pad), b)

        x = batch_norm(x)

        if nonlinearity is not None:
            x = nonlinearity(x)
        else:
            x = tf.nn.leaky_relu(x)
        return x


@add_arg_scope
def deconv_layer(x, num_filters, kernel_size, strides, pad="SAME", nonlinearity=None, counters=None, **kwargs):
    name = get_name('deconv', counters, kwargs.get("graph_name", ""))
    xshape = get_shape_as_list(x)
    if pad=='SAME':
        output_shape = [xshape[0], xshape[1]*strides[0], xshape[2]*strides[1], num_filters]
    else:
        output_shape = [xshape[0], xshape[1]*strides[0] + kernel_size[0]-1, xshape[2]*strides[1] + kernel_size[1]-1, num_filters]
    with tf.variable_scope(name):
        V = tf.get_variable(name = "V", shape = kernel_size + (num_filters, xshape[-1]), initializer=tf.random_normal_initializer(0, 0.05), dtype=tf.float32)
        b = tf.get_variable(name = "b", shape = [num_filters], initializer=tf.constant_initializer(0.), dtype=tf.float32)

        # calculate convolutional layer output
        x = tf.nn.conv2d_transpose(x, V, output_shape, (1,) + strides + (1,), padding=pad)
        x = tf.nn.bias_add(x, b)

        x = batch_norm(x)

        if nonlinearity is not None:
            x = nonlinearity(x)
        else:
            x = tf.nn.leaky_relu(x)
        return x

@add_arg_scope
def dense_layer(x, num_units, nonlinearity=None, counters=None, **kwargs):
    name = get_name('dense', counters, kwargs.get("graph_name", ""))
    with tf.variable_scope(name):
        xshape = get_shape_as_list(x)
        V = tf.get_variable(name="V", shape=[xshape[1], num_units], initializer=tf.random_normal_initializer(0,0.05), dtype=tf.float32)
        b = tf.get_variable(name="b", shape=[num_units], initializer=tf.constant_initializer(0.), dtype=tf.float32)

        # use weight normalization (Salimans & Kingma, 2016)
        x = tf.matmul(x, V) + b

        x = batch_norm(x)

        if nonlinearity is not None:
            x = nonlinearity(x)
        return x

@add_arg_scope
def batch_norm(x, training=True, epsilon=1e-5, counters=None, ema=None, store=None, **kwargs):
    """Code modification of http://stackoverflow.com/a/33950177"""
    name = get_name('batch_norm', counters, kwargs.get("graph_name", ""))
    with tf.variable_scope(name):
        shape = x.get_shape().as_list()

        beta = tf.get_variable("beta", [shape[-1]],
                            initializer=tf.constant_initializer(0.))
        gamma = tf.get_variable("gamma", [shape[-1]],
                            initializer=tf.random_normal_initializer(1., 0.02))

        if training:
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2,3][:len(shape)-1], name='moments')
            # ema_apply_op = ema.apply([batch_mean, batch_var])
            if store is not None:
                store["{}_ema_mean".format(name)] = ema.average(batch_mean)
                store["{}_ema_var".format(name)] = ema.average(batch_var)

            # with tf.control_dependencies([ema_apply_op]):
                mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            if store in None:
                pass
            mean, var = store["{}_ema_mean".format(name)], store["{}_ema_var".format(name)]

        return tf.nn.batch_norm_with_global_normalization(
                x, mean, var, beta, gamma, epsilon, scale_after_normalization=True)

def assert_finite(x):
    # pass
    assert tf.reduce_all(tf.is_finite(x)), "Non finite tensor: %s" % x
