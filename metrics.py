from nltk.translate import bleu_score
from rouge import Rouge


def get_bleu(targets, predictions):
    bleu_scores = 0
    smooth = bleu_score.SmoothingFunction()
    for i in range(len(predictions)):
        score = bleu_score.sentence_bleu(targets[i], predictions[i], smoothing_function=smooth.method7)
        bleu_scores += score

    return float(bleu_scores / len(targets))


def get_rouge(targets, predictions):
    rouge = Rouge()
    rouge_scores = 0
    for i in range(len(predictions)):
        score = rouge.get_scores(targets[i], predictions[i])[0]["rouge-l"]["f"]
        rouge_scores += score

    return float(rouge_scores / len(targets))


def get_distinct_1(sentences):
    unique_unigrams = []
    total_words = 0
    for s in sentences:
        total_words += len(s.split())
        for char in s:
            if char not in unique_unigrams:
                unique_unigrams.append(char)

    distinct_1 = float(len(unique_unigrams) / total_words)

    return distinct_1


def get_distinct_2(sentences):
    unique_bigrams = []
    total_words = 0
    for s in sentences:
        total_words += len(s.split())
        for i in range(len(s) - 1):
            if s[i:i + 2] not in unique_bigrams:
                unique_bigrams.append(s[i:i + 2])

    distinct_2 = float(len(unique_bigrams)/total_words)

    return distinct_2
