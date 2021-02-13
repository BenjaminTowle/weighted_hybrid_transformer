from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from dataloader import Dataloader
from config import Config
import pickle
from metrics import *


class Tfidf:
    def __init__(self):
        self.config = Config("tfidf")
        self.dataloader = Dataloader(self.config)
        self.vectorizer = TfidfVectorizer()

    def evaluate(self, metrics=None):
        if metrics is None:
            metrics = ["bleu", "rouge", "distinct1", "distinct2"]
            
        # Fit to data
        contexts, responses = self.dataloader.get_retrieval_candidates()
        X = self.vectorizer.fit_transform(contexts)

        # Evaluate on test data
        t_contexts, t_responses = self.dataloader.get_test_data()

        test_contexts = self.vectorizer.transform(t_contexts)
        predictions = []

        for context in test_contexts:
            cosine_similarities = linear_kernel(context, X).flatten()
            argsort = cosine_similarities.argsort()
            predictions.append(responses[argsort[-1]])

        pickle.dump((t_contexts, t_responses, predictions), open("Save/tfidf_test", "wb"))

        results = {}
        if "bleu" in metrics:
            results["bleu"] = get_bleu(t_responses, predictions)
        if "rouge" in metrics:
            results["rouge-l"] = get_rouge(t_responses, predictions)
        if "distinct1" in metrics:
            results["distinct-1"] = get_distinct_1(predictions)
        if "distinct2" in metrics:
            results["distinct-2"] = get_distinct_2(predictions)

        return results
