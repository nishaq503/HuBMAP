import numpy as np
import tensorflow as tf
from tensorflow import keras


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
    return tf.reduce_sum(tf.abs(x - y), axis=[-2, -1])


def embedding_loss(
        masks: tf.Tensor,
        embeddings: tf.Tensor,
):
    """ Computes the loss for the embeddings.

    Make it so that the distances between embeddings are proportional to the distances between masks.
    This loss is normalized by the dimensionality of the embedding and the mask.

    :param masks: tf.uint8 rank-3 tensor with shape (batch_size, image_height, image_width)
                        or rank-4 tensor with shape (batch_size, image_height, image_width, 1)
    :param embeddings: tf.float32 tensor with shape (batch_size, embedding_height, embedding_width, num_filters)
    :return: tf.float32 scalar Mean Absolute/Squared Error between embedding distances and mask distances.
    """
    # compute pairwise distances among masks and normalize by dimensionality
    masks_distances = tf.cast(_pairwise_distances_masks(masks), tf.float32)
    masks_distances /= tf.cast((tf.shape(masks)[1] * tf.shape(masks)[2]), tf.float32)

    # compute pairwise distances among flattened embeddings and normalize by dimensionality
    embeddings = tf.reshape(embeddings, shape=(tf.shape(embeddings)[0], -1))
    embeddings_distances = _pairwise_distances_embeddings(embeddings)
    embeddings_distances /= tf.cast(tf.shape(embeddings)[1], tf.float32)

    loss = tf.losses.mean_absolute_error(masks_distances, embeddings_distances)
    return loss


def dice_coef(true_masks, pred_masks, smooth: float = 1):
    true_masks = keras.backend.cast(true_masks, dtype=tf.float32)

    intersection = keras.backend.sum(true_masks * pred_masks, axis=[1, 2])
    union = keras.backend.sum(true_masks, axis=[1, 2]) + keras.backend.sum(pred_masks, axis=[1, 2])
    dice = keras.backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


def dice_loss(true_masks, pred_masks, smooth: float = 1):
    loss = 1. - dice_coef(true_masks, pred_masks, smooth)
    return loss


def ae_loss(true_images, pred_images):
    loss = tf.losses.mean_absolute_error(true_images, pred_images)
    return loss


# noinspection DuplicatedCode
def _test_pairwise_distances_embeddings():
    batch_size, embedding_dim = 32, 64
    embeddings = np.random.uniform(size=(batch_size, embedding_dim))
    results = np.zeros(shape=(batch_size, batch_size), dtype=np.float32)
    for i in range(batch_size):
        for j in range(batch_size):
            results[i, j] = np.sqrt(np.sum(np.square(embeddings[i] - embeddings[j])))

    tf_embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float32)
    tf_results = _pairwise_distances_embeddings(tf_embeddings)
    tf_results = np.asarray(tf_results)

    assert np.allclose(results, tf_results, 1e-3, 1e-3), f'results were different:\n{results}\n\n{tf_results}'
    return


# noinspection DuplicatedCode
def _test_pairwise_distances_masks():
    batch_size, image_size = 32, 512
    masks = np.random.randint(low=0, high=2, size=(batch_size, image_size, image_size), dtype=np.int32)
    results = np.zeros(shape=(batch_size, batch_size), dtype=np.int32)
    for i in range(batch_size):
        for j in range(batch_size):
            results[i, j] = 0 if i == j else np.sum(np.abs(masks[i] - masks[j]))

    tf_masks = tf.convert_to_tensor(masks)
    tf_results = _pairwise_distances_masks(tf_masks)
    tf_results = np.asarray(tf_results, dtype=np.int32)

    assert np.all(np.equal(results, tf_results)), f'results were different:\n{results}\n\n{tf_results}'
    return


# noinspection DuplicatedCode
def _test_embedding_loss():
    batch_size, image_size = 16, 512
    embedding_shape = (batch_size, 16, 16, 512)

    masks = np.random.randint(low=0, high=2, size=(batch_size, image_size, image_size), dtype=np.int32)
    masks_distances = np.zeros(shape=(batch_size, batch_size), dtype=np.int32)
    for i in range(batch_size):
        for j in range(batch_size):
            masks_distances[i, j] = 0 if i == j else np.sum(np.abs(masks[i] - masks[j]))
    masks_distances = masks_distances / (image_size * image_size)

    embeddings = np.random.uniform(size=embedding_shape)
    embeddings_distances = np.zeros(shape=(batch_size, batch_size), dtype=np.float32)
    for i in range(batch_size):
        for j in range(batch_size):
            embeddings_distances[i, j] = np.sqrt(np.sum(np.square(embeddings[i] - embeddings[j])))
    embeddings_distances = embeddings_distances / np.prod(embedding_shape[1:])

    results = np.abs(embeddings_distances - masks_distances)
    results = np.mean(results, axis=0)

    tf_masks = tf.convert_to_tensor(masks, dtype=tf.float32)
    tf_embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float32)
    tf_results = embedding_loss(tf_masks, tf_embeddings)
    assert np.allclose(results, tf_results), f'results were different:\n{results}\n\n{tf_results}'
    return


if __name__ == '__main__':
    _test_pairwise_distances_embeddings()
    _test_pairwise_distances_masks()
    _test_embedding_loss()
