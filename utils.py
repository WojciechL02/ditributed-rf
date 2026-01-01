import re
from collections import Counter
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from mpi4py import MPI


def clean_text(text: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned = cleaned.lower()
    return cleaned


def build_vocabulary(df) -> Counter:
    word_counts = Counter()

    print(f"Total rows loaded: {len(df)}")

    for idx, row in df.iterrows():
        text = str(row['text'])
        cleaned_text = clean_text(text)

        words = cleaned_text.split()
        word_counts.update(words)

        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1} rows...")

    print(f"Total rows processed: {len(df)}")
    return word_counts


def vocab_dimension_reduction(word_counts: Counter) -> set:
    vocabulary = {word for word, count in word_counts.items() if count > 1}
    return vocabulary


def filter_local_vocab(local_vocabulary: Counter, words_from_global_vocab: set[str]) -> set:
    vocabulary = {word for word, _ in local_vocabulary.items() if word in words_from_global_vocab}
    return vocabulary


def get_local_vocab(rank: int, comm: MPI.Intracomm, df: pd.DataFrame) -> set:
    # build and filter global vocabulary
    local_vocabulary = build_vocabulary(df)
    print(f"Rank: {rank}. Vocabulary size: {len(local_vocabulary)}")
    global_vocabulary = comm.allreduce(local_vocabulary, op=MPI.SUM)
    global_vocab_words = vocab_dimension_reduction(global_vocabulary)
    print(f"Rank: {rank}. Filtered global vocabulary size: {len(global_vocab_words)}")

    local_vocab_words = filter_local_vocab(local_vocabulary, global_vocab_words)
    return local_vocab_words


def get_dataset_by_rank(rank: int, base_path: str) -> pd.DataFrame:
    path = f"{base_path}_{rank}.csv"
    print("loading", path)
    df = pd.read_csv(
        path,
        header=None,
        names=["text", "label"],
        sep=",",
        quotechar='"',
        doublequote=True,
        on_bad_lines="warn",
    )
    return df


def count_vectorizer(text_series: pd.Series, vocab_list: list[str]) -> torch.Tensor:
    print(f"Vectorizing {len(text_series)} documents...")
    vocab_map = {word: i for i, word in enumerate(vocab_list)}

    rows = []
    cols = []
    token_pattern = re.compile(r"(?u)\b\w+\b")

    for row_idx, text in enumerate(text_series):
        tokens = token_pattern.findall(str(text).lower())
        seen_in_doc = set()
        for token in tokens:
            if token in vocab_map and token not in seen_in_doc:
                col_idx = vocab_map[token]
                rows.append(row_idx)
                cols.append(col_idx)
                seen_in_doc.add(token)

    values = np.ones(len(rows), dtype=np.int32)
    shape = (len(text_series), len(vocab_list))
    coo = sparse.coo_matrix((values, (rows, cols)), shape=shape)
    csr = coo.tocsr()

    return torch.sparse_csr_tensor(
        crow_indices=torch.from_numpy(csr.indptr).to(torch.int32),
        col_indices=torch.from_numpy(csr.indices).to(torch.int32),
        values=torch.from_numpy(csr.data).to(torch.int32),
        size=shape
    )
