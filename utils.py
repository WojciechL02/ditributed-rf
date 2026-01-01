import pickle
import numpy as np
import pandas as pd
import pickle
import torch
from itertools import islice


def vectorize(texts, vocab: dict):
    """Convert a text string into a numeric feature vector using the vocab."""
    num_texts = len(texts)
    vocab = dict(islice(vocab.items(), 100000))
    vocab_size = len(vocab)

    buffer = torch.zeros((num_texts, vocab_size), dtype=torch.uint8)

    for i, text in enumerate(texts):
        words = str(text.strip()).split()
        for word in words:
            if word in vocab:
                idx = vocab[word]
                buffer[i, idx] += 1
    return buffer

    # vec = np.zeros(len(vocab), dtype=int)
    # for word in text.split():
    #     if word in vocab:
    #         vec[vocab[word]] += 1  # count frequency
    # return vec


# with open("data/vocab.pkl", "rb") as f:
#     vocab = pickle.load(f)
#
# df = pd.read_csv("data/cleaned.csv")
# X = np.array([vectorize(text, vocab) for text in df["text"][:200000]])
# y = df["label"].values[:200000]
#
# print(X.shape)
# print(y.shape)