import tensorflow as tf

def _bit_to_float(string_batch: tf.Tensor):
    return tf.reshape(tf.math.floormod(tf.dtypes.cast(tf.bitwise.right_shift(
        tf.expand_dims(tf.io.decode_raw(string_batch, tf.uint8), 2),
        tf.reshape(tf.dtypes.cast(tf.range(7, -1, -1), tf.uint8), (1, 1, 8))
    ), tf.float32), 2), (tf.shape(string_batch)[0], -1))


def _parse_example_batch(example_batch):
    preprocessed_sample_columns = {
        "features": tf.io.VarLenFeature(tf.float32),
        "booleanFeatures": tf.io.FixedLenFeature((), tf.string, ""),
        "label": tf.io.FixedLenFeature((), tf.float32, -1)
    }
    samples = tf.io.parse_example(example_batch, preprocessed_sample_columns)
    dense_float = tf.sparse.to_dense(samples["features"])
    bits_to_float = _bit_to_float(samples["booleanFeatures"])
    return (
        tf.concat([dense_float, bits_to_float], 1),
        tf.reshape(samples["label"], (-1, 1))
    )

def build_dataset(file_pattern):
    return tf.data.Dataset.list_files(
        file_pattern
    ).interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).shuffle(
        buffer_size=2048
    ).cache(
    ).repeat(
    ).map(
        map_func=_parse_example_batch,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).batch(
        batch_size=2048,
        drop_remainder=True,
    ).prefetch(
        buffer_size=1
    )

#manually add one 
def test_dataset(file_pattern):
    ds = tf.data.Dataset.list_files(
        file_pattern
    )
    ds2 = ds.map(
        map_func=_parse_example_batch,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds3 = ds2.map(
        map_func=_parse_example_batch,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).batch(32)
    return ds3