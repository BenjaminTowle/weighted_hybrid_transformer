import pickle
import numpy as np
import tensorflow as tf
from config import Config
from Models.transformer import Transformer
from dataloader import Dataloader
from metrics import *
import os


class ModelNotFoundError(Exception):
    model_names = ["baseline", "weighted", "weighted_plus", "hybrid"]

    def __init__(self):
        message = """
        Requested model not in list of known models, model must be selected from 
        ["baseline", "weighted", "weighted_plus", "hybrid"]
        """
        super(ModelNotFoundError, self).__init__(message)


class Trainer:
    def __init__(self, model_name, config="auto"):
        """
        Trainer class is used for managing the end-to-end training procedure for the model, through to evaluation.
        :param model_name: string name of model
        :param config: config object containing details of model and training hyper-parameters.  Non-default parameters
        can be added by including them as dictionary, .e.g config={"model_size": 256}
        :raises ModelNotFoundError if the model_name given does not match the list of available models

        To run the training process:

            trainer = Trainer("baseline")
            trainer.train()
            # Some training happens
            # ...
            results = trainer.evaluate(["bleu", "rouge"])
            # {"bleu": 0.67, "rouge-l": 0.5}
        """
        if model_name not in ModelNotFoundError.model_names:
            raise ModelNotFoundError()
        
        if not os.path.isdir("Save"):
            os.mkdir("Save")

        self.config = Config(model_name)
        self.tokenizer = self.config.tokenizer
        if config != "auto":
            assert type(config) is dict
            for key, value in config.items():
                self.config.__setattr__(key, value)
        self.dataloader = Dataloader(self.config, multitask=self.config.multitask)
        self.transformer = Transformer(self.config)

        opt = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.transformer.compile(optimizer=opt)

    def train(self):
        """
        Trains the model according to the details provided in config.
        :return: None
        """
        dataset = tf.data.Dataset.from_generator(self.dataloader.generator, (np.int32, np.int32, np.int32))

        trailing_losses = {"generator": None, "retrieval": None, "reranker": None}
        validation_data = self.dataloader.get_validation_data()
        validation_losses = []

        for batch, inputs in enumerate(dataset.take(-1)):
            losses = self.transformer.train_step(inputs)
            if batch == 0:
                for key, value in losses.items():
                    trailing_losses[key] = value

            else:
                for key, value in losses.items():
                    trailing_losses[key] = self.config.momentum * trailing_losses[key] + (1 - self.config.momentum) * value

            # Reduce learning rate every DECAY_FREQ steps
            if (batch + 1) % self.config.decay_freq == 0:
                self.config.learning_rate *= self.config.decay_rate
                self.transformer.optimizer.__setattr__("learning_rate", self.config.learning_rate)

            if (batch + 1) % self.config.display_loss_freq == 0:
                print("Step: ", batch + 1)
                for key, value in trailing_losses.items():
                    if value is not None:
                        print(f"{key} loss: {np.round(value, 3)}")

                print(self.transformer.predict())
                self.transformer.encoder.save_weights(f"Save/{self.config.model_name}_encoder.h5")
                self.transformer.decoder.save_weights(f"Save/{self.config.model_name}_decoder.h5")

            if batch % self.config.validation_freq == 0:
                losses = self.transformer.validation_loss(validation_data)
                print("Step: ", batch + 1)
                for key, value in losses.items():
                    print(f"{key} loss: {np.round(value, 3)}")
                validation_losses.append(list(losses.values()))

            if (batch + 1) == self.config.num_steps:
                pickle.dump(validation_losses, open(f"Save/{self.config.model_name}_losses", "wb"))
                break

    def store_retrieval_candidates(self):
        """
        Stores a pickle file of the hidden states of the retrieval candidates and their text representations for use in
        inference.
        :return: None
        """
        contexts, responses = self.dataloader.get_retrieval_candidates()
        chunk_size = 100
        encoded_responses = []
        for i in range(0, self.config.retrieval_candidates, chunk_size):
            print(i)
            try:
                chunk_responses = responses[i:i + chunk_size]
                # Encode candidates
                candidates_encoded = self.tokenizer.encode_ids_with_bos_eos(chunk_responses)
                candidates = np.zeros([chunk_size, self.config.max_length])
                for j, c in enumerate(candidates_encoded):
                    for k in range(
                            len(c) - 1):  # Leave off EOS, as Encoder was only trained on responses with BOS token
                        candidates[j, k] = c[k]
                        # Truncate
                        if k == (self.config.max_length - 1):
                            break

                candidates = self.transformer.encoder.encode_responses(tf.constant(candidates))
                encoded_responses += (list(candidates.numpy()))
            except IndexError as e:
                print(e)
                print("error")
        encoded_responses = np.asarray(encoded_responses)
        pickle.dump(encoded_responses, open("response_vectors", "wb"))
        pickle.dump(responses, open("response_texts", "wb"))

    def evaluate(self, metrics=None):
        """
        Evaluates the model performance by first storing retrieval candidates if the model is a hybrid, then making
        predictions against the test set, and finally measuring the score according to the automated metrics specified.
        :param metrics: list of automated metrics.  Options are "bleu", "rouge", "distinct1", "distinct2".  All metrics
        will be used if none are specified
        :return: Dictionary of automated metrics and their associated scores
        """
        # Check if retrieval candidates exist
        if self.config.multitask:
            if not (os.path.isfile("Save/response_vectors") and os.path.isfile("Save/response_texts")):
                self.store_retrieval_candidates()

        # Check if model predictions exist
        if not os.path.isfile(f"Save/{self.config.model_name}_test"):
            contexts, responses = self.dataloader.get_test_data()
            self.transformer.test(contexts, responses)

        if metrics is None:
            metrics = ["bleu", "rouge", "distinct1", "distinct2"]
        _, responses, predictions = pickle.load(open(f"Save/{self.config.model_name}_test", "rb"))

        # Truncate responses to MAX_LENGTH tokens as this is what model made predictions against
        responses = self.tokenizer.encode_ids(responses)
        new_responses = []
        for r in responses:
            new_responses.append(r[:self.config.max_length])
        responses = self.tokenizer.decode_ids(new_responses)

        results = {}
        if "bleu" in metrics:
            results["bleu"] = get_bleu(responses, predictions)
        if "rouge" in metrics:
            results["rouge-l"] = get_rouge(responses, predictions)
        if "distinct1" in metrics:
            results["distinct-1"] = get_distinct_1(predictions)
        if "distinct2" in metrics:
            results["distinct-2"] = get_distinct_2(predictions)

        return results

