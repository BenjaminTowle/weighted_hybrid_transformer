import tensorflow as tf
from Models.multi_head_attention import MultiHeadAttention


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h, embedding, pes, multitask=False, activation="swish"):
        """
        Encoder component of the encoder-decoder model.
        :param vocab_size: size of vocabulary used for embeddings
        :param model_size: dimensionality of embeddings
        :param num_layers: number of layers
        :param h: number of attention heads
        :param embedding: 2D matrix of token vectors
        :param pes: 2D matrix of positional encoding
        :param multitask: Boolean whether or not the model is doing multitask training
        :param activation: string activation used for the hidden layers
        """
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.pes = pes
        self.multitask = multitask

        # One Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size, weights=[embedding], trainable=False)

        # num_layers Multi-Head Attention and Normalization layers
        self.attention = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        # num_layers FFN and Normalization layers
        self.dense_1 = [tf.keras.layers.Dense(model_size * 2, activation=activation) for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size, activation=activation) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        if multitask:
            # sentence embedding MLP
            self.mlp_context = tf.keras.layers.Dense(model_size, activation=activation)
            self.mlp_response = tf.keras.layers.Dense(model_size, activation=activation)

            # reranker MLP
            self.reranker_mlp = tf.keras.layers.Dense(model_size, activation=activation)

            # reranker cls
            self.classifier_head = tf.keras.layers.Dense(2, activation="softmax")

    def multitask_forward(self, inputs, masks=None):
        """
        Wraps around the standard call function for the model to include the additional calculations needed for the
        retrieval and reranking tasks.
        :param inputs: tuple of tensors containing contexts, responses and distractors
        :param masks: A dictionary containing a padding mask for each of the inputs
        :return: A dictionary containing the outputs and the losses from the retrieval and reranking tasks
        """
        if self.multitask:
            contexts, responses, distractors = inputs
        else:
            contexts = inputs

        if masks is not None:
            contexts_outputs = self(contexts, padding_mask=masks["contexts"])
            if self.multitask:
                responses_outputs = self(responses, padding_mask=masks["responses"])
                distractors_outputs = self(distractors, padding_mask=masks["distractors"])
        else:
            contexts_outputs = self(contexts)
            if self.multitask:
                responses_outputs = self(responses)
                distractors_outputs = self(distractors)

        outputs = {"encoder_outputs": contexts_outputs}
        if self.multitask:
            responses_hidden = tf.keras.layers.GlobalMaxPooling1D()(responses_outputs)
            distractors_hidden = tf.keras.layers.GlobalMaxPooling1D()(distractors_outputs)
            contexts_hidden = tf.keras.layers.GlobalMaxPooling1D()(contexts_outputs)
            contexts_hidden = self.mlp_context(contexts_hidden)
            responses_hidden = self.mlp_response(responses_hidden)
            distractors_hidden = self.mlp_response(distractors_hidden)

            # Normalise embeddings
            contexts_norm = tf.nn.l2_normalize(contexts_hidden, axis=-1)
            responses_norm = tf.nn.l2_normalize(responses_hidden, axis=-1)
            distractors_norm = tf.nn.l2_normalize(distractors_hidden, axis=-1)

            # Get cosine similarity
            true_similarity = 1.0 + tf.keras.losses.cosine_similarity(contexts_norm, responses_norm, axis=-1)
            false_similarity = 1.0 + tf.keras.losses.cosine_similarity(contexts_norm, distractors_norm, axis=-1)

            positive = 0.5 * tf.math.pow(true_similarity, 2)
            negative = 0.5 * tf.math.pow(tf.maximum(0.0, 1.0 - false_similarity), 2)

            contrastive_loss = tf.add(positive, negative)

            # Reranker
            reranker_true_inputs = tf.concat((contexts_hidden, responses_hidden), axis=-1)
            reranker_true_out = self.reranker_mlp(reranker_true_inputs)
            reranker_true_cls = self.classifier_head(reranker_true_out)
            reranker_true_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.ones(reranker_true_cls.shape[0], dtype=tf.float64), reranker_true_cls)

            reranker_false_inputs = tf.concat((contexts_hidden, distractors_hidden), axis=-1)
            reranker_false_out = self.reranker_mlp(reranker_false_inputs)
            reranker_false_cls = self.classifier_head(reranker_false_out)
            reranker_false_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.zeros(reranker_false_cls.shape[0], dtype=tf.float64), reranker_false_cls)

            reranker_loss = tf.add(reranker_true_loss, reranker_false_loss)

            outputs["contrastive_loss"] = contrastive_loss
            outputs["reranker_loss"] = reranker_loss

        return outputs

    def call(self, sequence, padding_mask=None):
        """
        Completes a forward pass of the model.
        :param sequence: a 2D tensor
        :param padding_mask: a 2D tensor that masks the sequence
        :return: the final hidden state from the model
        """
        # padding_mask will have the same shape as the input sequence
        # padding_mask will be used in the Decoder too
        # so we need to create it outside the Encoder
        embed_out = self.embedding(sequence)
        embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        embed_out += self.pes[:sequence.shape[1], :]

        sub_in = embed_out

        # We will have num_layers of (Attention + FFN)
        for i in range(self.num_layers):
            sub_out = self.attention[i](sub_in, sub_in, padding_mask)

            # Residual connection
            sub_out = sub_in + sub_out

            # Normalize the output
            sub_out = self.attention_norm[i](sub_out)

            # The FFN input is the output of the Multi-Head Attention
            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            # Add the residual connection
            ffn_out = ffn_in + ffn_out

            # Assign the FFN output to the next layer's Multi-Head Attention input
            sub_in = ffn_out

        # Return the result when done
        return ffn_out

    def rerank(self, contexts, candidates):
        """
        Scores a list of response candidates given a context through cross-encoding.
        :param contexts: a duplicated list of one context
        :param candidates: a 2D tensor of response candidates
        :return: a 1D tensor of scores for each respective candidate
        """
        contexts = self(contexts)
        candidates = self(candidates)

        contexts= tf.keras.layers.GlobalMaxPooling1D()(contexts)
        candidates = tf.keras.layers.GlobalMaxPooling1D()(candidates)

        contexts = self.mlp_context(contexts)
        candidates = self.mlp_response(candidates)

        inputs = tf.concat((contexts, candidates), axis=-1)
        inputs = self.reranker_mlp(inputs)
        scores = self.classifier_head(inputs)

        return scores

    def encode_responses(self, responses):
        """
        Encodes responses into a single hidden vector for each response for retrieval module.
        :param responses: 2D tensor of responses
        :return: 2D tensor of hidden states
        """
        responses = self(responses)
        responses = tf.keras.layers.GlobalMaxPooling1D()(responses)
        responses = self.mlp_response(responses)

        responses = tf.nn.l2_normalize(responses)

        return responses

    def encode_contexts(self, contexts):
        """
        Encodes contexts into a single hidden vector for each context for retrieval module
        :param contexts: 2D tensor of contexts
        :return: 2D tensor of hidden states
        """
        contexts = self(contexts)
        contexts = tf.keras.layers.GlobalMaxPooling1D()(contexts)
        contexts = self.mlp_context(contexts)

        contexts = tf.nn.l2_normalize(contexts)

        return contexts
