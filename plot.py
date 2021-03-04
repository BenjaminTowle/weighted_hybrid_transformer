import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import yticks
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import matplotlib.transforms as transforms
import numpy as np
import pickle
import os

if not os.path.isdir("Figures"):
    os.mkdir("Figures")

red = "#b85450"
blue = "#6c8ebf"

font = {'family' : 'DejaVu Serif',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)


def generator():
    data1 = pickle.load(open("Save/hybrid_losses", "rb"))
    data2 = pickle.load(open("Save/baseline_losses", "rb"))
    hybrid_losses = [loss[0] for loss in data1]
    baseline_losses = data2
    t = np.arange(len(data1))

    fig, ax = plt.subplots()
    plt.plot(t, hybrid_losses, label="Hybrid", color=red)
    plt.plot(t, baseline_losses, label="Baseline", color=blue)

    ax.set(xlabel="Steps ('000)", ylabel="Validation Loss")

    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    axes = plt.gca()
    axes.set_ylim([1, 6])
    yticks(np.arange(1, 6, 2))

    # Add horizontal line showing lowest point
    val = min(hybrid_losses)
    ax.axhline(y=val, color=red, linestyle="--")
    trans = transforms.blended_transform_factory(
    ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0,val, "{:.2f}".format(val), color=red, transform=trans,
        ha="right", va="center")
    val = min(baseline_losses)
    ax.axhline(y=val, color=blue, linestyle="--")
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, val, "{:.2f}".format(val), color=blue, transform=trans,
            ha="right", va="center")

    #plt.legend()
    plt.title("Generation")
    plt.tight_layout()
    plt.savefig("Figures/generator_loss.png")
    plt.show()


def retrieval():
    data1 = pickle.load(open("Save/hybrid_losses", "rb"))
    retrieval_losses = [loss[1] for loss in data1]
    t = np.arange(len(data1))
    fig, ax = plt.subplots()
    plt.plot(t, retrieval_losses, label="Hybrid", color=red)

    ax.set(xlabel="Steps ('000)", ylabel="Validation Loss")

    # Add horizontal line showing lowest point
    val = min(retrieval_losses)
    ax.axhline(y=val, color=red, linestyle="--")
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, val, "{:.2f}".format(val), color=red, transform=trans,
            ha="right", va="center")

    axes = plt.gca()
    axes.set_ylim([0.2, 0.3])
    yticks(np.arange(0.2, 0.3, 0.05))
    #plt.legend()
    plt.title("Retrieval")
    plt.tight_layout()

    plt.savefig("Figures/retrieval_loss.png")
    plt.show()


def reranker():
    data1 = pickle.load(open("Save/hybrid_losses", "rb"))
    rerank_losses = [loss[2] for loss in data1]
    t = np.arange(len(data1))
    fig, ax = plt.subplots()
    ax.set(xlabel="Steps ('000)", ylabel="Validation Loss")
    plt.plot(t, rerank_losses, label="Hybrid", color=red)

    # Add horizontal line showing lowest point
    val = min(rerank_losses)
    ax.axhline(y=val, color=red, linestyle="--")
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, val, "{:.2f}".format(val), color=red, transform=trans,
            ha="right", va="center")

    axes = plt.gca()
    axes.set_ylim([1.30, 1.50])
    yticks(np.arange(1.30, 1.50, 0.10))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))

    #plt.legend()
    plt.title("Reranking")
    plt.tight_layout()

    plt.savefig("Figures/reranker_loss.png")
    plt.show()


generator()
retrieval()
reranker()
