import numpy as np
import pandas as pd
from .pool import Pool
from .asktell import AskTellFewShotTopk
from .llm_model import GaussDist

from typing import *
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim.fit import fit_gpytorch_torch
import torch
from langchain.embeddings import OpenAIEmbeddings
from sklearn.manifold import Isomap


class AskTellGPR(AskTellFewShotTopk):
    def __init__(self, n_components=2, pool=None, cache_path=None, **kwargs):
        super().__init__(**kwargs)
        self._set_regressor()
        self.examples = []
        self._embedding = OpenAIEmbeddings()
        self._embeddings_cache = self._get_cache(cache_path)
        self.isomap = Isomap(n_components=n_components)
        self.pool = pool
        if self.pool is not None:
            self._initialize_isomap()

    def _initialize_isomap(self):
        pool_embeddings = self._query_cache(self.pool._available)
        self.isomap.fit(pool_embeddings)

    def _get_cache(self, cache_path=None):
        try:
            cache = pd.read_csv(cache_path)
            print(f"Loaded cache from {cache_path}.")
        except:
            print("Cached embeddings not found. Creating new cache table.")
            cache = pd.DataFrame({"x": [], "embedding": []})
        return cache

    def save_cache(self, cache_path):
        self._embeddings_cache.to_csv(cache_path, index=False)

    def _query_cache(self, X):
        in_cache = self._embeddings_cache["x"].to_list()
        not_in_cache = np.setdiff1d(X, in_cache)
        new_embeddings = []
        if not_in_cache.size > 0:
            new_embeddings = self._embedding.embed_documents(not_in_cache.tolist(), 250)
        self._embeddings_cache = pd.concat(
            [
                self._embeddings_cache,
                pd.DataFrame({"x": not_in_cache, "embedding": new_embeddings}),
            ],
            ignore_index=True,
        )
        embedding = [
            self._embeddings_cache[self._embeddings_cache["x"] == xi][
                "embedding"
            ].to_list()[0]
            for xi in X
        ]

        if len(embedding) != len(X):
            raise ValueError(
                (
                    "Embedding length does not match X length."
                    "Something went wrong on caching."
                )
            )

        return embedding

    def _set_regressor(self):
        self.likelihood = GaussianLikelihood()
        self.regressor = None

    def _predict(self, X):
        if len(X) == 0:
            raise ValueError("X is empty")
        embedding = self._query_cache(X)
        embedding_isomap = self.isomap.transform(embedding)
        results = []
        with torch.no_grad():
            self.regressor.eval()
            self.likelihood.eval()
            means = self.likelihood(self.regressor(torch.tensor(embedding_isomap))).mean
            stds = self.likelihood(
                self.regressor(torch.tensor(embedding_isomap))
            ).variance.sqrt()
        results = [GaussDist(mean.item(), std.item()) for mean, std in zip(means, stds)]
        return results, 0

    def _train(self, X, y):
        embedding = self._query_cache(X)
        embedding = np.array(embedding)
        if self.pool is None:
            embedding_isomap = self.isomap.fit_transform(embedding)
        else:
            embedding_isomap = self.isomap.transform(embedding)
        train_x = torch.tensor(embedding_isomap)
        train_y = torch.tensor(list(map(float, y))).unsqueeze(-1).double()
        self.regressor = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(self.regressor.likelihood, self.regressor)
        fit_gpytorch_torch(mll)

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

    def tell(
        self, x: str, y: float, alt_ys: Optional[List[float]] = None, train=True
    ) -> None:
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
            self.inv_llm = self._setup_inv_llm(self._model, self._temperature)
            self._ready = True

        self.examples.append(example_dict)

        if train:
            self._train(
                [
                    self.prompt.format(
                        x=ex["x"],
                        y_name=self._y_name,
                    )
                    for ex in self.examples
                ],
                [ex["y"] for ex in self.examples],
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
