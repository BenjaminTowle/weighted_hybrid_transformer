import tensorflow as tf
from Models.multi_head_attention import MultiHeadAttention


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h, embedding, pes, activation="swish"):
        """
        Decoder component of the encoder-decoder model.
        :param vocab_size: size of vocabulary used for embeddings
        :param model_size: dimensionality of embeddings
        :param num_layers: number of layers
        :param h: number of attention heads
        :param embedding: 2D matrix of token vectors
        :param pes: 2D matrix of positional encoding
        :param activation: string activation used for the hidden layers
        """
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.pes = pes
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size, weights=[embedding], trainable=False)
        self.attention_bot = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_bot_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_mid_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(model_size * 4, activation=activation) for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size, activation=activation) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, encoder_output, padding_mask=None):
        """
        Foward pass of the decoder model.
        :param sequence: 2D tensor of decoder inputs
        :param encoder_output: 2D tensor of encoder final hidden state
        :param padding_mask: 2D tensor of mask for encoder padding
        :return: 3D tensor of logits from output layer
        """
        # EMBEDDING AND POSITIONAL EMBEDDING
        embed_out = self.embedding(sequence)
        embed_out += self.pes[:sequence.shape[1], :]

        bot_sub_in = embed_out
        seq_len = bot_sub_in.shape[1]

        for i in range(self.num_layers):
            # BOTTOM MULTIHEAD SUB LAYER
            look_left_only_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            bot_sub_out = self.attention_bot[i](bot_sub_in, bot_sub_in, look_left_only_mask)

            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out = self.attention_mid[i](mid_sub_in, encoder_output, padding_mask)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out

            logits = self.dense(ffn_out)

        return logits
