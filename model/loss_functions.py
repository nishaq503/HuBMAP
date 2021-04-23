import numpy as np
import tensorflow as tf


def _pairwise_distances_embeddings(embeddings: tf.Tensor, squared: bool = False):
    """ Pairwise distances between flat embeddings.

    :param embeddings: tf.float32 tensor with shape (batch_size, embedding_dim)
    :param squared: whether to use squared distances
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

    if not squared:
        # sqrt(0) has inf gradient so avoid this by adding a small epsilon
        mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
        distances = tf.sqrt(distances + mask * 1e-16)
        # Correct for the epsilon added set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _pairwise_distances_masks(masks: tf.Tensor):
    """ Pairwise distances between square masks.

    :param masks: tf.int32 tensor with shape (batch_size, image_height, image_width)
    :return: tf.int32 tensor with shape (batch_size, batch_size) of pairwise distances between masks.
    """
    x = tf.expand_dims(masks, 0)
    y = tf.expand_dims(masks, 1)
    return tf.reduce_sum(tf.abs(x - y), axis=[-2, -1])


def embedding_loss(
        masks: tf.Tensor,
        embeddings: tf.Tensor,
        squared: bool = False,
):
    """ Computes the loss for the embeddings.

    Make it so that the distances between embeddings are proportional to the distances between masks.
    This loss is normalized by the size of the mask.

    :param masks: tf.int32 tensor with shape (batch_size, image_height, image_width)
    :param embeddings: tf.float32 tensor with shape (batch_size, embedding_dim)
    :param squared: bool. whether to use squared distances and squared differences between distances.
    :return: tf.float32 scalar Mean Absolute/Squared Error between embedding distances and mask distances.
    """
    masks_distances = tf.cast(_pairwise_distances_masks(masks), tf.float32)
    embeddings_distances = _pairwise_distances_embeddings(embeddings, squared)

    differences = tf.abs(masks_distances - embeddings_distances)
    if squared:
        differences = tf.square(differences)

    size = tf.shape(masks)
    factor = tf.cast(size[1] * size[2], dtype=tf.float32)
    factor = tf.sqrt(factor) * tf.cast(tf.shape(embeddings)[1], tf.float32)
    return tf.reduce_mean(differences) / factor


def masking_loss(
        true_masks: tf.Tensor,
        predicted_masks: tf.Tensor,
        squared: bool = False,
        smoothed: bool = True,
):
    """ Mean Absolute/Squared Error loss between true masks and predicted masks.

    :param true_masks: True segmentation masks for images.
                        tf.int32 tensor with shape (batch_size, image_height, image_width)
    :param predicted_masks: predicted segmentation masks for images
                        tf.float32 tensor with shape (batch_size, image_height, image_width, 1)
    :param squared: Whether to use squared error
    :param smoothed: Whether to smooth the predicted masks with a gaussian filter.
    :return:
    """
    pass


# noinspection DuplicatedCode
def _test_pairwise_distances_embeddings():
    batch_size = 32
    embedding_dim = 64
    embeddings = np.random.uniform(size=(batch_size, embedding_dim))
    results = np.zeros(shape=(batch_size, batch_size), dtype=np.float32)
    for i in range(batch_size):
        for j in range(batch_size):
            results[i, j] = np.sqrt(np.sum(np.square(embeddings[i] - embeddings[j])))

    tf_embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float32)
    tf_results = _pairwise_distances_embeddings(tf_embeddings)
    tf_results = np.asarray(tf_results)

    assert np.allclose(results, tf_results), f'results were different:\n{results}\n\n{tf_results}'
    return


# noinspection DuplicatedCode
def _test_pairwise_distances_masks():
    batch_size = 32
    masks = np.random.randint(low=0, high=2, size=(batch_size, 256, 256), dtype=np.int32)
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
    batch_size, embedding_dim, image_size = 16, 64, 1024

    masks = np.random.randint(low=0, high=2, size=(batch_size, image_size, image_size), dtype=np.int32)
    masks_distances = np.zeros(shape=(batch_size, batch_size), dtype=np.int32)
    for i in range(batch_size):
        for j in range(batch_size):
            masks_distances[i, j] = 0 if i == j else np.sum(np.abs(masks[i] - masks[j]))

    embeddings = np.random.uniform(size=(batch_size, embedding_dim))
    embeddings_distances = np.zeros(shape=(batch_size, batch_size), dtype=np.float32)
    for i in range(batch_size):
        for j in range(batch_size):
            embeddings_distances[i, j] = np.sqrt(np.sum(np.square(embeddings[i] - embeddings[j])))

    results = np.abs(embeddings_distances - masks_distances)
    results = np.mean(results) / (image_size * embedding_dim)

    tf_masks = tf.convert_to_tensor(masks, dtype=tf.float32)
    tf_embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float32)
    tf_results = embedding_loss(tf_masks, tf_embeddings)
    assert np.allclose(results, tf_results), f'results were different:\n{results}\n\n{tf_results}'
    return


def _test_masking_loss():
    pass


if __name__ == '__main__':
    # _test_pairwise_distances_embeddings()
    # _test_pairwise_distances_masks()
    _test_embedding_loss()
