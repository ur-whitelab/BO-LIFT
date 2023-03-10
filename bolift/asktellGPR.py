import numpy as np
import pandas as pd
from .pool import Pool
from .asktell import AskTellFewShotTopk
from .llm_model import GaussDist

from typing import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, PairwiseKernel, RBF
from langchain.embeddings import OpenAIEmbeddings

def cosine_similarity(X, Y=None, dense_output=True, gamma=None):
    if Y is None:
        Y = X
    X = X.reshape(1, -1)
    Y = Y.reshape(1, -1)
    dot = np.dot(X, Y.T)
    norms_X = np.linalg.norm(X, axis=1).reshape(-1, 1)
    norms_Y = np.linalg.norm(Y, axis=1).reshape(1, -1)
    norms = norms_X * norms_Y
    sim = dot / norms
    return sim if dense_output else np.diag(sim)


class AskTellGPR(AskTellFewShotTopk):
    def __init__(self, cache_path=None, **kwargs):
        super().__init__(**kwargs)
        # self._selector_k = 0
        self._set_regressor()
        self.examples = []
        self._embedding = OpenAIEmbeddings()
        self._embeddings_cache = self._get_cache(cache_path)
        # self._embeddings_cache = pd.DataFrame({'x':[], 'embedding':[]})
        # self._embeddings_cache = {}


    # I need to get the query from a persisted file
    def _get_cache(self, cache_path=None):
        try:
            cache = pd.read_csv(cache_path)
            print(f"Loaded cache from {cache_path}.")
        except:
            print("Cached embeddings not found. Creating new cache table.")
            cache = pd.DataFrame({'x':[], 'embedding':[]})
        return cache


    def save_cache(self, cache_path):
        self._embeddings_cache.to_csv(cache_path, index=False)


    def _query_cache(self, X):
        in_cache = self._embeddings_cache['x'].to_list()
        # in_cache = self._embeddings_cache.keys()
        not_in_cache = np.setdiff1d(X, in_cache)
        new_embeddings = []
        if not_in_cache.size > 0:
            new_embeddings = self._embedding.embed_documents(not_in_cache.tolist(), 250)
        self._embeddings_cache = pd.concat(
            [
                self._embeddings_cache, 
                pd.DataFrame({'x':not_in_cache, 'embedding':new_embeddings})
            ], ignore_index=True)
        embedding = [
            self._embeddings_cache[self._embeddings_cache['x']==xi]['embedding'].to_list()[0] 
            for xi in X
            ]
        # self._embeddings_cache = {**self._embeddings_cache, **dict(zip(not_in_cache, new_embeddings))}
        # print('update cache', ti-time.time())
        # embedding = [
        #     self._embeddings_cache[xi] 
        #     for xi in X
        # ]

        if len(embedding) != len(X):
            raise ValueError(('Embedding length does not match X length.'
                              'Something went wrong on caching.'))
            # print(self._embeddings_cache[self._embeddings_cache['x'].isin(X)])
            # print(X)

        return embedding


    def _set_regressor(self):
        cosine_kernel = PairwiseKernel(metric=cosine_similarity)
        constant_kernel = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed")
        cos_kernel = constant_kernel * cosine_kernel
        self.regressor = GaussianProcessRegressor(
            kernel=cos_kernel, n_restarts_optimizer=2,
            alpha=0.001, normalize_y=False
            # kernel=RBF(length_scale=1e-3, length_scale_bounds=(1e-10, 1e1)),
            # kernel=cos_kernel,
            # n_restarts_optimizer=2,
            # normalize_y=True,
        )


    def _predict(self, X):
        if(len(X) == 0):
            raise ValueError("X is empty")
        embedding=self._query_cache(X)
        # embedding = np.array(self._embedding.embed_documents(X, 250))
        results = []
        means, stds = self.regressor.predict(embedding, return_std=True)
        results = [GaussDist(mean, std) for mean, std in zip(means, stds)]
        return results, 0

    def _train(self, X, y):
        embedding=self._query_cache(X)
        # embedding = np.array(self._embedding.embed_documents(X, 250))
        self.regressor.fit(embedding, list(map(float, y)))


    def ask(
        self,
        possible_x: Union[Pool, List[str]],
        aq_fxn: str = "upper_confidence_bound",
        k: int = 1,
        inv_filter: int = 16,
        _lambda: float = 0.5,
    ) -> Tuple[List[str], List[float], List[float]]:
        # just have this here to override default
        return super().ask(possible_x, aq_fxn, k, 0, _lambda)


    def tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> None:
        """Tell the optimizer about a new example."""
        example_dict, inv_example = self._tell(x, y, alt_ys)
        # we want to have example
        # to initialize prompts, so send it
        if not self._ready:
            self.prompt = self._setup_prompt(
                None, self._prompt_template, self._suffix, self._prefix
            )
            self.inv_prompt = self._setup_inverse_prompt(inv_example)
            self.llm = self._setup_llm(self._model, self._temperature)
            self._ready = True

        self.examples.append(example_dict)

        self._train(
            [self.prompt.format(
                x=ex["x"],
                y_name=self._y_name,
                )
             for ex in self.examples
            ], 
            [ex["y"] for ex in self.examples]
        )

    def predict(self, x: str) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """Predict the probability distribution and values for a given x.

        Args:
            x: The x value(s) to predict.
        Returns:
            The probability distribution and values for the given x.

        """
        if not isinstance(x, list):
            x = [x]
        if not self._ready:
            # special zero-shot
            self.prompt = self._setup_prompt(
                None, self._prompt_template, self._suffix, self._prefix
            )
            self.inv_prompt = self._setup_inverse_prompt(None)
            self.llm = self._setup_llm(self._model)
            self._ready = True

        if self._selector_k is not None:
            # have to update this until my PR is merged
            self.prompt.example_selector.k = min(self._example_count, self._selector_k)

        queries = [
            self.prompt.format(
                x=self.format_x(x_i),
                y_name=self._y_name,
            )
            for x_i in x
        ]
        results, tokens = self._predict(queries)
        self.tokens_used += tokens

        # compute mean and standard deviation
        if len(x) == 1:
            return results[0]
        return results