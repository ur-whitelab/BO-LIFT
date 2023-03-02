"""utilities for building and selecting from a pool"""
from typing import List, Any, Callable
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


class Pool:
    """Class for sampling from pool of possible data points

    Example:
        >>> pool = Pool(['a', 'b', 'c', 'd', 'e'])
        >>> pool.sample(3)
        ['a', 'd', 'c']
        >>> pool.choose('a')
        >>> pool.sample(3)
        ['b', 'c', 'd']
        >>> pool.approx_sample('a', 3)
        ['b', 'c', 'd']
    """

    def __init__(self, pool: List[Any], formatter: Callable = lambda x: str(x)) -> None:
        if type(pool) is not list:
            raise TypeError("Pool must be a list")
        self._pool = pool
        self._selected = []
        self._available = pool[:]
        self.format = formatter
        self._db = FAISS.from_texts(
            [formatter(x) for x in pool],
            OpenAIEmbeddings(),
            metadatas=[dict(data=p) for p in pool],
        )

    def sample(self, n: int) -> List[str]:
        """Sample n items from the pool"""
        if n > len(self._available):
            raise ValueError("Not enough items in pool")
        samples = np.random.choice(self._available, size=n, replace=False)
        return samples

    def choose(self, x: str) -> None:
        """Choose a specific item from the pool"""
        if x not in self._available:
            raise ValueError("Item not in pool")
        self._selected.append(x)
        self._available.remove(x)

    def approx_sample(self, x: str, k: int) -> None:
        """Given an approximation of x, return k similar"""
        # want to select extra, then remove previously chosen
        _k = k + len(self._selected)
        docs = self._db.max_marginal_relevance_search(x, k=_k, fetch_k=5 * _k)
        docs = [d.metadata["data"] for d in docs]
        # remove previously chosen
        docs = [d for d in docs if d not in self._selected]
        # select k
        return docs[:k]

    def reset(self) -> None:
        """Reset the pool"""
        self._selected = []
        self._available = self._pool[:]

    def __len__(self) -> int:
        return len(self._pool)

    def __repr__(self) -> str:
        return f"Pool of {len(self)} items with {len(self._selected)} selected"

    def __str__(self) -> str:
        return f"Pool of {len(self)} items with {len(self._selected)} selected"

    def __iter__(self):
        return iter(self._available)
