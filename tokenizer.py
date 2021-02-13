from bpemb import BPEmb
import numpy as np


class Tokenizer(BPEmb):
    def __init__(self, **kwargs):
        """
        Tokenizer to tokenize data using byte-pair encoding
        :param kwargs: language, vocab size, embedding dimensionality
        """
        super(Tokenizer, self).__init__(**kwargs)

    def encode_pad_sequences(self, contexts, responses, max_length=20):
        """
        Takes list of context and response ids and returns numpy arrays with 0 padding
        :param contexts: list of context ids
        :param responses: list of response ids
        :param max_length: maximum length for context/response.  Anything longer than this is truncated
        :return: 2D numpy array of encoder_inputs, 2D numpy array of decoder_inputs, 2D numpy array of decoder_outputs
        """
        num_samples = len(contexts)
        # Convert to byte-pair encodings
        contexts_ids = self.encode_ids(contexts)
        responses_ids = self.encode_ids_with_bos_eos(responses)

        # Convert to array and truncate any sentences above max length
        # BPEmb uses 1 for bos, 2 for eos, and 0 for padding
        encoder_inputs_contexts = np.zeros([num_samples, max_length], dtype=np.int32)
        decoder_inputs = np.zeros([num_samples, max_length], dtype=np.int32)
        decoder_outputs = np.zeros([num_samples, max_length], dtype=np.int32)
        for j, context in enumerate(contexts_ids):
            for k, id_ in enumerate(context):
                encoder_inputs_contexts[j, k] = id_

                # Apply truncation
                if k == (max_length - 2):
                    break

        for j, response in enumerate(responses_ids):
            for k in range(len(response) - 1):
                decoder_inputs[j, k] = response[k]
                decoder_outputs[j, k] = response[k + 1]

                # Apply truncation
                if k == (max_length - 2):
                    break

        return encoder_inputs_contexts, decoder_inputs, decoder_outputs
