import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()

from src.model import get_model, vae_loss
from src.utils import merge

BATCH_SIZE = 256


def train(num_filters, num_encoder_layers, num_decoder_layers, num_epochs):
    optimizer = tf.train.AdamOptimizer(1e-3)
    container = tf.contrib.eager.EagerVariableStore()

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(np.float32)
    train_images /= 256  # rescale to 0-1.

    buffer_size = train_images.shape[0]
    im_size = train_images.shape[-2]
    dataset = tf.data.Dataset.from_tensor_slices(train_images) \
        .shuffle(buffer_size) \
        .batch(BATCH_SIZE)
    encoder, decoder = get_model(
        im_size=im_size, num_channels=train_images.shape[-1],
        num_filters=num_filters, num_encoder_layers=num_encoder_layers,
        num_dense_encoder_layers=1, num_decoder_layers=num_decoder_layers,
        num_dense_decoder_layers=1)
    generate_images = get_image_generator(decoder, 16, num_filters*im_size*im_size)

    num_iterations = buffer_size//BATCH_SIZE
    for j in range(num_epochs):
        epoch_losses = []
        for i, x in enumerate(dataset):
            with tf.GradientTape() as gr_tape, container.as_default():
                h_means, h_std = encoder(x)
                logits = decoder(h_means, h_std)

                loss = vae_loss(x, logits, h_means, h_std)

            gradients = gr_tape.gradient(
                loss,
                container.trainable_variables()
            )
            optimizer.apply_gradients(
                zip(gradients, container.trainable_variables())
            )
            epoch_losses.append(loss)
            if i % (num_iterations//10) == 0:
                print("*", end="", flush=True)
        print("\nAverage loss over epoch %d: %s" % (j, float(tf.reduce_mean(epoch_losses))))
        sample_images = generate_images(h_means[:16,...],h_std[:16,...])
        sample_image = merge(sample_images, [4, 4])[:,:,0]
        plt.imsave("./samples/warm_sample_img_{}.jpg".format(j), sample_image, cmap = 'gray')
        sample_images = generate_images()
        sample_image = merge(sample_images, [4, 4])[:,:,0]
        plt.imsave("./samples/raw_sample_img_{}.jpg".format(j), sample_image, cmap = 'gray')



def get_image_generator(decoder, num_images, num_latent_vars):
    def generator(h_means=None, h_std=None):
        if h_means is None or h_std is None:
            h_means = np.zeros([num_images, num_latent_vars])
            h_std = np.ones([num_images, num_latent_vars])
        logits = decoder(h_means, h_std)
        return tf.nn.sigmoid(logits)
    return generator


if __name__ == "__main__":
    train(4,1,1,100)