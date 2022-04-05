import numpy as np
import tensorflow as tf


#UpperTri was taken from Basenji https://github.com/calico/basenji
class UpperTri(tf.keras.layers.Layer):
  ''' Unroll matrix to its upper triangular portion.'''
  def __init__(self, diagonal_offset=2, **kwargs):
    super(UpperTri, self).__init__()
    self.diagonal_offset = diagonal_offset

  def call(self, inputs):
    seq_len = inputs.shape[1]
    output_dim = inputs.shape[-1]

    if type(seq_len) == tf.compat.v1.Dimension:
      seq_len = seq_len.value
      output_dim = output_dim.value

    triu_tup = np.triu_indices(seq_len, self.diagonal_offset)
    triu_index = list(triu_tup[0]+ seq_len*triu_tup[1])
    unroll_repr = tf.reshape(inputs, [-1, seq_len**2, output_dim])
    return tf.gather(unroll_repr, triu_index, axis=1)

  def get_config(self):
    config = super().get_config().copy()
    config['diagonal_offset'] = self.diagonal_offset
    return config