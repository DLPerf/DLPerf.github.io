import tensorflow as tf
import random
from pathlib import glob

_keys_to_map = {
    'd': tf.FixedLenFeature([], tf.string),  # data
    's': tf.FixedLenFeature([], tf.int64),   # score
}

def init_tfrecord_dataset():
  files_train = glob.glob(DIR_TFRECORDS + '*.tfrecord')
  random.shuffle(files_train)

  with tf.name_scope('tfr_iterator'):
    ds = tf.data.TFRecordDataset(files_train)      # define data from randomly ordered files
    ds = ds.shuffle(buffer_size=10000)             # select elements randomly from the buffer
    ds = ds.map(_parser)                           # map them based on tfrecord format
    ds = ds.batch(BATCH_SIZE, drop_remainder=True) # group elements in batch (remove batch of less than BATCH_SIZE)
    ds = ds.repeat()                               # iterate infinitely 

    return ds.make_initializable_iterator()        # initialize the iterator


def iterator_to_data(iterator):
  """Creates a part of the graph which reads the raw data from an iterator and transforms it to a 
  data ready to be passed to model.

  Args:
    iterator      - iterator. Created by `init_tfrecord_dataset`

  Returns:
    data_board      - (BATCH_SIZE, 8, 8, 24) you can think about as NWHC for images.
    data_flags      - (BATCH_SIZE, 10)
    combined_score  - (BATCH_SIZE,)
  """

  b = tf.constant((128, 64, 32, 16, 8, 4, 2, 1), dtype=tf.uint8, name='unpacked_const')

  with tf.name_scope('tfr_parse'):
    with tf.name_scope('packed_data'):
      next_element = iterator.get_next()
      data_packed, score_int = next_element
      score = tf.cast(score_int, tf.float64, name='score_float')

    # https://stackoverflow.com/q/45454470/1090562
    with tf.name_scope('data_unpacked'):
      data_unpacked = tf.reshape(tf.mod(tf.to_int32(tf.decode_raw(data_packed, tf.uint8)[:,:,None] // b), 2), [BATCH_SIZE, 1552], name='data_unpack')

    with tf.name_scope('score'):
      with tf.name_scope('is_mate'):
        score_is_mate = tf.cast(tf.squeeze(tf.slice(data_unpacked, [0, 1546], [BATCH_SIZE, 1])), tf.float64, name='is_mate')
      with tf.name_scope('combined'):
        combined_score = (1 - score_is_mate) * VALUE_A * tf.tanh(score / VALUE_K) + score_is_mate * tf.sign(score) * (VALUE_A + (1 - VALUE_A) / (VALUE_B - 1) * tf.reduce_max(tf.stack([tf.zeros(BATCH_SIZE, dtype=tf.float64), VALUE_B - tf.abs(score)]), axis=0))


    with tf.name_scope('board'):
      with tf.name_scope('reshape_layers'):
        data_board = tf.reshape(tf.slice(data_unpacked, [0, 0], [BATCH_SIZE, 8 * 8 * 24]), [BATCH_SIZE, 8, 8, 24], name='board_reshape')

      with tf.name_scope('combine_layers'):  
        data_board = tf.cast(tf.stack([
          data_board[:,:,:, 0],
          data_board[:,:,:, 4],
          data_board[:,:,:, 8],
          data_board[:,:,:,12],
          data_board[:,:,:,16],
          data_board[:,:,:,20],
          - data_board[:,:,:, 1],
          - data_board[:,:,:, 5],
          - data_board[:,:,:, 9],
          - data_board[:,:,:,13],
          - data_board[:,:,:,17],
          - data_board[:,:,:,21],
          data_board[:,:,:, 2],
          data_board[:,:,:, 6],
          data_board[:,:,:,10],
          data_board[:,:,:,14],
          data_board[:,:,:,18],
          data_board[:,:,:,22],
          - data_board[:,:,:, 3],
          - data_board[:,:,:, 7],
          - data_board[:,:,:,11],
          - data_board[:,:,:,15],
          - data_board[:,:,:,19],
          - data_board[:,:,:,23],
        ], axis=3), tf.float64, name='board_compact')

    with tf.name_scope('flags'):
      data_flags = tf.cast(tf.slice(data_unpacked, [0, 1536], [BATCH_SIZE, 10]), tf.float64, name='flags')

  return data_board, data_flags, combined_score