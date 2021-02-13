# weighted_hybrid_transformer
An encoder-decoder transformer model which trains on the Open Subtitles dataset.  There are four variants of the transformer that can be used "hybrid", "weighted", "weighted_plus" and "baseline".  The "hybrid" model is a multitask transformer that performs response retrieval and reranking tasks in addition to response generation.  The "weighted" and "weighted plus" models apply a modifier to cross-entropy loss for high frequency words to reduce the impact of their overrepresentation in the dataset and theoretically make the model's generation less bland.  The "baseline" is the regualar transformer without any of the above.  The specific transformer architecture used is largely a smaller version of Vaswani et al.'s (2017) ["Attention is all you need"](https://arxiv.org/abs/1706.03762) transformer.

## Dependencies:
* tensorflow >= 2.3.1
* nltk >= 3.5
* datasets >= 1.2.0
* rouge >= 1.0.0

The files can be operated through the Trainer class.  Specific details about the model hyper-parameters can be changed when the Trainer class is instantiated.  See below:

```python
trainer = Trainer("baseline", config={"num_layers": 3})
trainer.train()
# Some training happens ...
# ...
# Now, for evaluation
results = trainer.evaluate(metrics=["bleu"])
# {"bleu": 0.6534}
```
