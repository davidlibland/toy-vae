import numpy as np

def merge(images, shape):
    n, h, w, c = tuple(map(int, images.shape))
    assert int(np.prod(shape)) == n, \
        "Incompatible shapes: %s != %s" % (np.prod(shape), n)
    img = np.zeros((h * shape[0], w * shape[1], c))

    for idx, image in enumerate(images):
        i = idx % shape[1]
        j = idx // shape[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img