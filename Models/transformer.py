import tensorflow as tf
import numpy as np
from scipy.special import softmax
import pickle
import sys
from Models.decoder import Decoder
from Models.encoder import Encoder


class Transformer(tf.keras.Model):
    def __init__(self, config):
        """
        Transformer model that wraps over separate Encoder and Decoder classes, as well as provided the methods used
        during training and inference.
        :param config: config object containing specification for how to build the encoder-decoder architecture
        """
        super(Transformer, self).__init__()
        self.config = config
        pes = []
        for i in range(self.config.max_length):
            pes.append(self.positional_embedding(i, self.config.model_size))
        pes = np.concatenate(pes, axis=0)
        pes = tf.constant(pes, dtype=tf.float32)
        # Original implementation of Vaswani et al.'s [2017] Attention is All you Need Encoder-Decoder Transformer
        # leveraging https://trungtran.io/2019/04/29/create-the-transformer-with-tensorflow-2-0/, onto which
        # substantial modifications have been made accomodate the hybrid architecture.
        self.encoder = Encoder(self.config.vocab_size, self.config.model_size, self.config.num_layers, self.config.h,
                               self.config.tokenizer.vectors, pes=pes, multitask=self.config.multitask)
        self.decoder = Decoder(self.config.vocab_size, self.config.model_size, self.config.num_layers, self.config.h,
                               self.config.tokenizer.vectors, pes=pes)

        self.multitask = self.config.multitask
        self.stopwords = self.config.stopwords
        self.stopword_l = self.config.stopword_l

    def train_step(self, inputs):
        """
        Called during each update step.
        :param inputs: A tuple of 2D tensors of encoder_inputs, decoder_inputs, and decoder_outputs
        :return: A dictionary mapping task loss to values
        """
        encoder_inputs_contexts, decoder_inputs, decoder_outputs = inputs
        with tf.GradientTape() as tape:
            if self.multitask:
                # Create inputs for retrieval and re-ranking tasks
                encoder_inputs_responses = tf.identity(decoder_inputs)
                encoder_inputs_distractors = tf.identity(decoder_inputs)
                tf.random.shuffle(encoder_inputs_distractors)

                # Create padding mask to block attention on padding (PAD has id 0)
                contexts_padding_mask = 1 - tf.cast(tf.equal(encoder_inputs_contexts, 0), dtype=tf.float32)
                responses_padding_mask = 1 - tf.cast(tf.equal(encoder_inputs_responses, 0), dtype=tf.float32)
                distractors_padding_mask = 1 - tf.cast(tf.equal(encoder_inputs_distractors, 0), dtype=tf.float32)

                # Add additional dimension to mask (batch_size, 1, seq_len)
                contexts_padding_mask = tf.expand_dims(contexts_padding_mask, axis=1)
                responses_padding_mask = tf.expand_dims(responses_padding_mask, axis=1)
                distractors_padding_mask = tf.expand_dims(distractors_padding_mask, axis=1)
                masks = {"contexts": contexts_padding_mask, "responses": responses_padding_mask,
                         "distractors": distractors_padding_mask}

                encoder_output = self.encoder.multitask_forward(
                    [encoder_inputs_contexts, encoder_inputs_responses, encoder_inputs_distractors], masks)
                retrieval_loss = encoder_output["contrastive_loss"]
                reranker_loss = encoder_output["reranker_loss"]

                decoder_output = self.decoder(decoder_inputs, encoder_output["encoder_outputs"], masks["contexts"])

                generator_loss = self.loss_func(decoder_outputs, decoder_output)
                losses = {"generator": np.mean(generator_loss.numpy()),
                          "retrieval": np.mean(retrieval_loss.numpy()),
                          "reranker": np.mean(reranker_loss.numpy())}
                loss = generator_loss + retrieval_loss + reranker_loss

            else:
                encoder_outputs = self.encoder(encoder_inputs_contexts)
                pred = self.decoder(decoder_inputs, encoder_outputs)
                generator_loss = self.loss_func(decoder_outputs, pred)
                losses = {"generator": np.mean(generator_loss.numpy())}
                loss = generator_loss

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return losses

    def load_weights(self, folder):
        self.encoder.load_weights(f"{folder}/{self.config.model_name}_encoder.h5")
        self.decoder.load_weights(f"{folder}/{self.config.model_name}_decoder.h5")

    def predict(self, test_source_text=None, top_k=None, return_probabilities=False):
        """
        Used at inference, to take an input sentence and produce a response.
        :param test_source_text: A string specifying the context; a random dummy context will be used if not provided
        :param top_k: An integer that determines number of candidates for each resampling in top_k decoding.  If not
        provided, greedy decoding will be used
        :param return_probabilities: Boolean, set to True to return tuple containing response and probability, otherwise
        will return only response
        :return: A string response or a tuple of string response and float probability
        """
        # If test sentence is not provided
        # randomly pick up one from the below
        if test_source_text is None:
            test_source_text = np.random.choice([
                "hello, how are you?",
                "what is your name?",
                "hi there",
                "what's up?"
            ])
            print(test_source_text)

        # Tokenize the test sentence to obtain source sequence
        test_source_seq = self.config.tokenizer.encode_ids([test_source_text])

        # Convert to tensor and truncate
        en_output = self.encoder(tf.constant(test_source_seq)[:, :self.config.max_length])

        de_input = tf.constant([[1]], dtype=tf.int64)
        out_words = []
        probability = 1.

        while True:
            de_output = self.decoder(de_input, en_output)

            if top_k is None:
                # Take the last token as the predicted token
                new_word = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)

                out_words.append(int(new_word.numpy()[0][0]))

                # The next input is a new sequence
                # contains both the input sequence and the predicted token
                de_input = tf.concat((de_input, new_word), axis=1)

            else:
                # Select top_k tokens
                p = []
                array = de_output.numpy()
                argsort = np.argsort(array[0, -1, :])
                candidates = argsort[-top_k:]
                for k in range(top_k):
                    p.append(array[0, -1, candidates[k]])
                new_word = np.random.choice(candidates, p=softmax(p))
                probability *= softmax(array[0, -1, :])[int(new_word)]

                out_words.append(int(new_word))

                # The next input is a new sequence
                # contains both the input sequence and the predicted token
                de_input = tf.concat((de_input, tf.cast(tf.constant([[new_word]]), tf.int64)), axis=1)

            # End if hitting <end> or length exceeds 14
            if out_words[-1] == 2 or len(out_words) >= 14:
                break

        if return_probabilities:
            return self.config.tokenizer.decode_ids(out_words), probability

        else:
            return self.config.tokenizer.decode_ids(out_words)

    def test(self, contexts, responses):
        """
        Generates predictions based on the contexts provided, and stores as a pickle file.
        :param contexts: list of strings
        :param responses: list of strings
        :return: None
        """
        if self.config.multitask:
            # Load retrieval candidates
            try:
                retrieval_vectors = pickle.load(open("Save/response_vectors", "rb"))
                retrieval_texts = pickle.load(open("Save/response_texts", "rb"))
            except FileNotFoundError as e:
                print(e)
                print("Must initialise retrieval candidates first")
                sys.exit()

        predictions = []
        generative_usage = 0
        for context in contexts:
            generated_responses = []
            if not self.config.multitask:
                # Generate response candidates
                probs = []
                for _ in range(self.config.num_generated):
                        response, prob = self.predict(test_source_text=context, top_k=5, return_probabilities=True)
                        generated_responses.append(response)
                        probs.append(prob)
                argmax = int(np.argmax(np.asarray(probs)))

            else:
                for _ in range(self.config.num_generated):
                    generated_responses.append(self.predict(context, top_k=5))

                # Rerank candidates
                # Encode context and duplicate
                context_encoded = self.config.tokenizer.encode_ids([context])[0]
                context_encoded = context_encoded[:self.config.max_length]
                context_encoded = [context_encoded for _ in range(self.config.num_generated + self.config.num_retrieved)]

                # Context encode
                retrieval_encode_context = self.encoder.encode_contexts(tf.constant([context_encoded[0]])).numpy()
                scores = np.dot(retrieval_encode_context[0], retrieval_vectors.T)
                argsort = np.argsort(scores)
                retrieval_candidates = argsort[-self.config.num_retrieved:]

                for i in range(self.config.num_retrieved):
                    generated_responses.append(retrieval_texts[int(retrieval_candidates[i])])

                # Encode candidates
                candidates_encoded = self.config.tokenizer.encode_ids_with_bos_eos(generated_responses)
                candidates = np.zeros([len(generated_responses), self.config.max_length])
                for j, c in enumerate(candidates_encoded):
                    for k in range(len(c) - 1):  # Leave off EOS, as Encoder was only trained on responses with BOS token
                        candidates[j, k] = c[k]

                        if k == (self.config.max_length - 1):
                            break

                context_encoded = tf.constant(context_encoded)
                candidates = tf.constant(candidates)

                scores = self.encoder.rerank(context_encoded, candidates)
                argmax = np.argmax(scores.numpy()[:, 0])

            print("=====================")
            print(context)
            pred = generated_responses[int(argmax)]
            if int(argmax) <= 9:
                generative_usage += 1
            predictions.append(pred)
            print(pred)
            print("====================")

        pickle.dump((contexts, responses, predictions), open(f"Save/{self.config.model_name}_test", "wb"))
        print(generative_usage)

    def validation_loss(self, inputs):
        """
        Completes a forward pass on the model with validation data and calculates the loss
        :param inputs: tuple containing 2D tensors of encoder_inputs, decoder_inputs, and decoder_outputs
        :return: A dictionary mapping task losses to values
        """
        encoder_inputs_contexts, decoder_inputs, decoder_outputs = inputs
        if self.multitask:
            # Create inputs for retrieval and re-ranking tasks
            encoder_inputs_responses = tf.identity(decoder_inputs)
            encoder_inputs_distractors = tf.identity(decoder_inputs)
            tf.random.shuffle(encoder_inputs_distractors)
            encoder_outputs = self.encoder.multitask_forward([encoder_inputs_contexts, encoder_inputs_responses,
                                                            encoder_inputs_distractors])
            retrieval_loss = encoder_outputs["contrastive_loss"]
            reranker_loss = encoder_outputs["reranker_loss"]

            pred = self.decoder(decoder_inputs, encoder_outputs["encoder_outputs"])
            generator_loss = self.loss_func(decoder_outputs, pred)
            losses = {"generator": np.mean(generator_loss.numpy()),
                      "retrieval": np.mean(retrieval_loss.numpy()),
                      "reranker": np.mean(reranker_loss.numpy())}

        else:
            encoder_outputs = self.encoder(encoder_inputs_contexts)
            pred = self.decoder(decoder_inputs, encoder_outputs)
            generator_loss = self.loss_func(decoder_outputs, pred)
            losses = {"generator": np.mean(generator_loss.numpy())}

        return losses

    def loss_func(self, targets, logits):
        """
        Calculates cross-entropy loss by applying a mask to the padding tokens, and weighted the stopwords if the
        specific model requires it.
        :param targets: 2D tensor of target indices
        :param logits: 3D tensor of predictions
        :return: cross-entropy loss
        """
        crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        mask = tf.cast(tf.math.logical_not(tf.math.equal(targets, 0)), tf.float64)
        if self.stopwords is not None:
            for id_ in self.stopwords:
                tmp_mask = tf.cast(tf.math.equal(targets, id_), tf.float64) * -self.stopword_l
                mask = tf.add(mask, tmp_mask)
        loss = crossentropy(targets, logits, sample_weight=mask)

        return loss

    @staticmethod
    def positional_embedding(pos, model_size):
        """
        Encodes position according to the positional embedding used in Vaswani et al's [2017] Attention is all you need.
        :param pos: integer position in the sentence
        :param model_size: integer dimensionality of the embedding
        :return: 2D numpy array
        """
        PE = np.zeros((1, model_size))
        for i in range(model_size):
            if i % 2 == 0:
                PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
            else:
                PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))

        return PE
