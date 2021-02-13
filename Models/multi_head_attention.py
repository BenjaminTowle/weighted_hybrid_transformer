import tensorflow as tf


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.query_size = model_size // h
        self.key_size = model_size // h
        self.value_size = model_size // h
        self.h = h
        self.wq = [tf.keras.layers.Dense(self.query_size) for _ in range(h)]
        self.wk = [tf.keras.layers.Dense(self.key_size) for _ in range(h)]
        self.wv = [tf.keras.layers.Dense(self.value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, query, value, mask=None):
        # query has shape (batch, query_len, model_size)
        # value has shape (batch, value_len, model_size)
        heads = []
        for i in range(self.h):
            score = tf.matmul(self.wq[i](query), self.wk[i](value), transpose_b=True)
            score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
            # score has shape (batch, query_len, value_len
            # mask must be broadcastable to (batch, query_len, value_len)
            if mask is not None:
                score *= mask

                # assign masked positions to -1e9
                # so that their values after softmax are zero
                score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)

            alignment = tf.nn.softmax(score, axis=2)
            # alignment has shape (batch, query_len, value_len

            head = tf.matmul(alignment, self.wv[i](value))
            # head has shape (batch, query_len, value_size
            heads.append(head)

        # Concatenate all the attention heads
        # so that the last dimension summed up to model size
        heads = tf.concat(heads, axis=-1)
        heads = self.wo(heads)

        # heads has shape (batch, query_len, model_size)
        return heads
