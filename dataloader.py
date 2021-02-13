from datasets import load_dataset
import datasets
import tensorflow as tf

datasets.temp_seed(1)


class Dataloader:
    def __init__(self, config, multitask=False):
        """
        The Dataloader class is used to generate and provide data from the Open Subtitles dataset.
        :param config: config object containing specification about how many training samples should be used etc.
        :param multitask: Boolean specifies whether multitask training is being done
        """
        self.data = load_dataset("open_subtitles", lang1="en", lang2="is")
        self.multitask = multitask
        self.config = config

    def get_validation_data(self):
        """
        Fetches the validation dataset
        :return: 2D tensor of encoder_inputs, 2D tensor of decoder_inputs, 2D tensor of decoder_outputs
        """
        validation_data = self.data["train"][self.config.num_samples: self.config.num_samples + self.config.validation_samples + 1]["translation"]
        contexts = []
        responses = []
        for j in range(len(validation_data) - 1):
            contexts.append(validation_data[j]["en"])
            responses.append(validation_data[j + 1]["en"])

        encoder_inputs_contexts, decoder_inputs, decoder_outputs = self.config.tokenizer.encode_pad_sequences(contexts, responses)

        return tf.constant(encoder_inputs_contexts), tf.constant(decoder_inputs), tf.constant(decoder_outputs)

    def generator(self):
        """
        A generator function that continuously provides batches of training data
        :return: 2D tensor of encoder_inputs, 2D tensor of decoder_inputs, 2D tensor of decoder_outputs
        """
        i = 0

        while True:
            utterances = self.data["train"][i: i+self.config.batch_size+1]["translation"]
            contexts = []
            responses = []
            for j in range(len(utterances)-1):
                contexts.append(utterances[j]["en"])
                responses.append(utterances[j+1]["en"])
            i += self.config.batch_size
            if i >= self.config.num_samples:
                i = 0

            encoder_inputs_contexts, decoder_inputs, decoder_outputs = \
                self.config.tokenizer.encode_pad_sequences(contexts, responses)

            yield encoder_inputs_contexts, decoder_inputs, decoder_outputs

    def get_test_data(self):
        """
        Fetches lists of contexts and responses from test data
        :return: list of context strings, list of response strings
        """
        test_data = \
            self.data["train"][self.config.num_samples + 500: self.config.num_samples + 500 + self.config.test_samples + 1][
                "translation"]
        contexts = []
        responses = []
        for j in range(len(test_data) - 1):
            contexts.append(test_data[j]["en"])
            responses.append(test_data[j + 1]["en"])

        return contexts, responses

    def get_retrieval_candidates(self):
        """
        Fetches lists of contexts and responses for retrieval model
        :return: list of context strings, list of response strings
        """
        # Index retrieval candidates
        retrieval_data = self.data["train"][:self.config.retrieval_candidates]["translation"]
        contexts = []
        responses = []
        for j in range(len(retrieval_data) - 1):
            contexts.append(retrieval_data[j]["en"])
            responses.append(retrieval_data[j + 1]["en"])

        return contexts, responses
