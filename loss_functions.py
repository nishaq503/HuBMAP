import numpy
import tensorflow as tf
from tensorflow import keras

import utils


def _pairwise_distances_embeddings(embeddings: tf.Tensor):
    """ Pairwise distances between flat embeddings.

    :param embeddings: tf.float32 tensor with shape (batch_size, embedding_dim)
    :return: tf.float32 tensor with shape (batch_size, batch_size) of pairwise distances.
    """
    # Get the dot product between all embeddings
    dot_product = tf.linalg.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding.
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of floating-point errors, some distances might be negative so we set everything >= 0.0
    distances = tf.maximum(distances, 0.0)
    # sqrt(0) has inf gradient so avoid this by adding a small epsilon
    mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
    distances = tf.sqrt(distances + mask * 1e-16)
    # Correct for the epsilon added set the distances on the mask to be exactly 0.0
    distances = distances * (1.0 - mask)
    return distances


def _pairwise_distances_masks(masks: tf.Tensor):
    """ Pairwise distances between square masks.

    :param masks: tf.uint8 tensor with shape (batch_size, image_height, image_width)
    :return: tf.float32 tensor with shape (batch_size, batch_size) of pairwise distances between masks.
    """
    x = tf.cast(tf.expand_dims(masks, 0), tf.float32)
    y = tf.cast(tf.expand_dims(masks, 1), tf.float32)
    distances = tf.reduce_sum(tf.abs(x - y), axis=[2, 3])
    return distances


def embedding_loss(masks: tf.Tensor, embeddings: tf.Tensor):
    """ Computes the loss for the embeddings.

    Make it so that the distances between embeddings are proportional to the distances between masks.
    This loss is normalized by the dimensionality of the embedding and the mask.

    :param masks: tf.float32 rank-3 tensor with shape (batch_size, image_height, image_width)
                          or rank-4 tensor with shape (batch_size, image_height, image_width, 1)
    :param embeddings: tf.float32 tensor with shape (batch_size, embedding_height, embedding_width, num_filters)
    :return: tf.float32 scalar Mean Absolute/Squared Error between embedding distances and mask distances.
    """
    half_batch_size = int(tf.shape(masks)[0] / 2)
    embeddings = tf.reshape(embeddings, shape=(tf.shape(embeddings)[0], -1))

    # Pairwise distances among the glom-masks need to be proportional to the
    # pairwise distances among the glom-embeddings.
    glom_masks, glom_embeddings = masks[:half_batch_size], embeddings[:half_batch_size]
    glom_masks_distances = _pairwise_distances_masks(glom_masks)
    glom_masks_distances /= tf.cast((tf.shape(masks)[1] * tf.shape(masks)[2]), tf.float32)
    glom_embeddings_distances = _pairwise_distances_embeddings(glom_embeddings)
    glom_embeddings_distances /= tf.cast((tf.shape(embeddings)[1]), tf.float32)
    glom_embeddings_distances += tf.cast(1e-16, tf.float32)
    ratio = tf.divide(glom_masks_distances, glom_embeddings_distances)
    glom_loss = tf.math.reduce_std(ratio, axis=1)

    # pairwise distances among blank embeddings need to be zero.
    blank_embeddings = embeddings[half_batch_size:]
    blank_embeddings_distances = _pairwise_distances_embeddings(blank_embeddings)
    blank_loss = tf.reduce_sum(blank_embeddings_distances, axis=1)

    loss = tf.concat([glom_loss, blank_loss], axis=0)
    return loss


def dice_coef(true_masks, pred_masks, smooth: float = 1e-8):
    true_masks = tf.cast(true_masks, dtype=tf.float32)
    pred_masks = tf.cast(pred_masks, dtype=tf.float32)

    intersection = tf.reduce_sum(true_masks * pred_masks, axis=[1, 2])
    true_sum = tf.reduce_sum(true_masks, axis=[1, 2])
    pred_sum = tf.reduce_sum(pred_masks, axis=[1, 2])
    dice = (2. * intersection + smooth) / (true_sum + pred_sum + smooth)
    return dice


def dice_loss(true_masks, pred_masks, smooth: float = 1e-8):
    loss = 1. - dice_coef(true_masks, pred_masks, smooth)
    return loss


def dice_bce(true_masks, pred_masks, smooth: float = 1e-8):
    bce_loss = keras.losses.binary_crossentropy(true_masks, pred_masks)
    bce_loss = tf.reduce_mean(bce_loss, axis=1)
    return dice_loss(true_masks, pred_masks, smooth) + bce_loss


# noinspection DuplicatedCode
def _test_pairwise_distances_embeddings():
    batch_size = utils.GLOBALS['batch_size']
    embedding_dim = int(numpy.prod((16, 16, 192)))
    embeddings = numpy.random.uniform(size=(batch_size, embedding_dim))
    results = numpy.zeros(shape=(batch_size, batch_size), dtype=numpy.float32)
    for i in range(batch_size):
        for j in range(batch_size):
            results[i, j] = numpy.sqrt(numpy.sum(numpy.square(embeddings[i] - embeddings[j])))

    tf_embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float32)
    tf_results = _pairwise_distances_embeddings(tf_embeddings)
    tf_results = numpy.asarray(tf_results)

    assert numpy.allclose(results, tf_results, 1e-3, 1e-3), f'results were different:\n{results}\n\n{tf_results}'
    return


# noinspection DuplicatedCode
def _test_pairwise_distances_masks():
    batch_size, image_size = utils.GLOBALS['batch_size'], utils.GLOBALS['tile_size']
    masks = numpy.random.randint(low=0, high=2, size=(batch_size, image_size, image_size), dtype=numpy.int8)
    results = numpy.zeros(shape=(batch_size, batch_size), dtype=numpy.float32)
    for i in range(batch_size):
        for j in range(batch_size):
            results[i, j] = 0 if i == j else numpy.sum(numpy.abs(masks[i] - masks[j]))

    masks = numpy.asarray(masks, dtype=numpy.uint8)

    tf_masks = tf.convert_to_tensor(masks, dtype=tf.float32)
    tf_results = _pairwise_distances_masks(tf_masks)
    tf_results = numpy.asarray(tf_results, dtype=numpy.float32)

    assert numpy.all(numpy.equal(results, tf_results)), f'results were different:\n{results}\n\n{tf_results}'
    return


# noinspection DuplicatedCode
def _test_embedding_loss():
    batch_size, image_size = utils.GLOBALS['batch_size'], utils.GLOBALS['tile_size']
    embedding_shape = (batch_size, 16, 16, 192)

    masks = numpy.random.randint(low=0, high=2, size=(batch_size, image_size, image_size), dtype=numpy.int8)
    embeddings = numpy.random.uniform(size=embedding_shape)

    tf_masks = tf.convert_to_tensor(masks, dtype=tf.float32)
    tf_embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float32)
    tf_results = embedding_loss(tf_masks, tf_embeddings)
    assert numpy.all(tf_results >= 0), f'some results were negative:\n{tf_results}'
    return


if __name__ == '__main__':
    # _test_pairwise_distances_embeddings()
    # _test_pairwise_distances_masks()
    _test_embedding_loss()
