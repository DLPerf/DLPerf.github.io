import tensorflow as tf

dataset = tf.data.TFRecordDataset(data_file)
dataset = dataset.prefetch(buffer_size=batch_size*10)
dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
dataset = dataset.repeat(num_epochs)
dataset = dataset.batch(batch_size)

features, labels = dataset.make_one_shot_iterator().get_next()    
logits = tf.feature_column.linear_model(features=features, feature_columns=columns, cols_to_vars=cols_to_vars)
train_op = ...

with tf.Session() as sess:
    sess.run(train_op)