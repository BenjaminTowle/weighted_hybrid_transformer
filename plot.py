import matplotlib.pyplot as plt
from matplotlib.pyplot import yticks
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import numpy as np
import pickle
import os

if not os.path.isdir("Figures"):
    os.mkdir("Figures")


def generator():
    data1 = pickle.load(open("Save/hybrid_losses", "rb"))
    data2 = pickle.load(open("Save/baseline_losses", "rb"))
    hybrid_losses = [loss[0] for loss in data1]
    baseline_losses = data2
    t = np.arange(len(data1))

    fig, ax = plt.subplots()
    plt.plot(t, hybrid_losses, label="Hybrid")
    plt.plot(t, baseline_losses, label="Base")

    ax.set(xlabel="Steps ('000)", ylabel="Validation loss (CE)", yscale="log")

    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    yticks([2, 4, 6, 8, 10])
    ax.grid()
    plt.legend()

    plt.savefig("Figures/generator_loss.png")
    plt.show()


def retrieval():
    data1 = pickle.load(open("Save/hybrid_losses", "rb"))
    retrieval_losses = [loss[1] for loss in data1]
    t = np.arange(len(data1))
    fig, ax = plt.subplots()
    plt.plot(t, retrieval_losses, label="Retrieval")

    ax.set(xlabel="Steps ('000)", ylabel="Validation loss")

    ax.grid()
    plt.legend()

    plt.savefig("Figures/retrieval_loss.png")
    plt.show()


def reranker():
    data1 = pickle.load(open("Save/hybrid_losses", "rb"))
    rerank_losses = [loss[2] for loss in data1]
    t = np.arange(len(data1))
    fig, ax = plt.subplots()
    ax.set(xlabel="Steps ('000)", ylabel="Validation loss")
    plt.plot(t, rerank_losses, label="Reranker")

    axes = plt.gca()
    axes.set_ylim([1.3, 1.5])
    yticks(np.arange(1.3, 1.5, 0.02))

    ax.grid()
    plt.legend()

    plt.savefig("Figures/reranker_loss.png")
    plt.show()

generator()
retrieval()
reranker()
