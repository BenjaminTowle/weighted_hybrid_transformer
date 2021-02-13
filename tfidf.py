from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from dataloader import Dataloader
from config import Config
import pickle


class Tfidf:
    def __init__(self):
        self.config = Config("tfidf")
        self.dataloader = Dataloader(self.config)
        self.vectorizer = TfidfVectorizer()

    def evaluate(self):
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

        pickle.dump((t_contexts, t_responses, predictions), open("tfidf", "wb"))
