import numpy as np
import pandas as pd
from .pool import Pool
from .asktell import AskTellFewShot, QuantileTransformer
from .llm_model import GaussDist

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_community.vectorstores import FAISS, Chroma

from typing import *
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim.fit import fit_gpytorch_mll_torch
import torch
from langchain_openai import OpenAIEmbeddings
from sklearn.manifold import Isomap
from openai import OpenAI


class AskTellGPR(AskTellFewShot):
    def __init__(
        self, n_components=32, pool=None, cache_path=None, n_neighbors=5, **kwargs
    ):
        super().__init__(**kwargs)
        self._selector_k = None  # Forcing exemple_selector to not build context
        self._set_regressor()
        self.examples = []
        self._embedding = OpenAIEmbeddings()
        self._embeddings_cache = self._get_cache(cache_path)
        self.isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
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
        """
        Queries embeddings from cache; fetches missing embeddings via OpenAI API in batches.

        Parameters:
            X (list of str): Input data for which embeddings are needed.

        Returns:
            List of embeddings corresponding to X.
        """
        in_cache = self._embeddings_cache["x"].to_list()
        not_in_cache = np.setdiff1d(X, in_cache).tolist()

        not_in_cache = [
            str(i) for i in not_in_cache if isinstance(i, str) and i.strip()
        ]

        batch_size = 5
        new_embeddings = []

        if len(not_in_cache) > 0:
            print(
                f"Processing {len(not_in_cache)} new items in {len(not_in_cache) // batch_size + 1} batches..."
            )

            client = OpenAI()

            for i in range(0, len(not_in_cache), batch_size):
                batch = not_in_cache[i : i + batch_size]

                try:
                    response = client.embeddings.create(
                        input=batch,
                        model="text-embedding-ada-002",
                        encoding_format="float",
                    )

                    batch_embeddings = [data.embedding for data in response.data]
                    new_embeddings.extend(batch_embeddings)

                except Exception as e:
                    print(f"❌ Error processing batch {i//batch_size + 1}: {e}")
                    continue

            if new_embeddings:
                self._embeddings_cache = pd.concat(
                    [
                        self._embeddings_cache,
                        pd.DataFrame({"x": not_in_cache, "embedding": new_embeddings}),
                    ],
                    ignore_index=True,
                )

        else:
            print("✅ No new items to process; all embeddings found in cache.")

        embedding = []
        for xi in X:
            result = self._embeddings_cache[self._embeddings_cache["x"] == xi][
                "embedding"
            ].to_list()
            if result:
                embedding.append(result[0])
            else:
                raise ValueError(
                    f"❌ Embedding for '{xi}' not found in cache after update."
                )

        if len(embedding) != len(X):
            raise ValueError(
                "❌ Embedding length does not match X length. Caching issue detected."
            )

        return embedding

    # def _query_cache(self, X):
    #     in_cache = self._embeddings_cache["x"].to_list()
    #     not_in_cache = np.setdiff1d(X, in_cache)
    #     new_embeddings = []
    #     # print("length in not in cache:",len(not_in_cache))
    #     if not_in_cache.size > 0:
    #         print(f"Processing {len(not_in_cache)} items...")
    #         client = OpenAI()
    #         for i in not_in_cache:
    #             response = client.embeddings.create(
    #                 input=i,
    #                 model= "text-embedding-ada-002" #"text-embedding-3-small"
    #             )
    #             new_embeddings.append(response.data[0].embedding)
    #     else:
    #         print("No items in not_in_cache to process.")

    #     #print(len(new_embeddings[0]),not_in_cache,type(X))
    #     self._embeddings_cache = pd.concat(
    #         [
    #             self._embeddings_cache,
    #             pd.DataFrame({"x": not_in_cache, "embedding": new_embeddings}),
    #         ],
    #         ignore_index=True,
    #     )
    #     print("Make it to 1")
    #     embedding = [
    #         self._embeddings_cache[self._embeddings_cache["x"] == xi][
    #             "embedding"
    #         ].to_list()[0]
    #         for xi in X
    #     ]

    #     print("Make it to 2")

    #     if len(embedding) != len(X):
    #         raise ValueError(
    #             (
    #                 "Embedding length does not match X length."
    #                 "Something went wrong on caching."
    #             )
    #         )
    #     print("Make it to 3")

    #     return embedding

    def _set_regressor(self):
        self.likelihood = GaussianLikelihood()
        self.regressor = None

    def _setup_prompt(
        self,
        example: Dict,
        prompt_template: Optional[PromptTemplate] = None,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> FewShotPromptTemplate:
        if prefix is None:
            prefix = (
                "The following are correctly answered questions. "
                "Each answer is numeric and ends with ###\n"
            )
        if prompt_template is None:
            prompt_template = PromptTemplate(
                input_variables=["x", "y", "y_name"],
                template="Q: Given {x}, what is {y_name}?\nA: {y}###\n\n",
            )
            if suffix is not None:
                raise ValueError(
                    "Cannot provide suffix if using default prompt template."
                )
            suffix = "Q: Given {x}. What is {y_name}?\nA: "
        elif suffix is None:
            raise ValueError("Must provide suffix if using custom prompt template.")
        # test out prompt
        if example is not None:
            prompt_template.format(**example)
            examples = [example]
        # TODO: make fake example text
        else:
            examples = []
        example_selector = None
        if self._selector_k is not None:
            if len(examples) == 0:
                raise ValueError("Cannot do zero-shot with selector")
            sim_selector = (
                SemanticSimilarityExampleSelector
                if self.cos_sim
                else MaxMarginalRelevanceExampleSelector
            )
            example_selector = sim_selector.from_examples(
                [example],
                OpenAIEmbeddings(),
                FAISS,
                k=self._selector_k,
            )
        return FewShotPromptTemplate(
            examples=examples if example_selector is None else None,
            example_prompt=prompt_template,
            example_selector=example_selector,
            suffix=suffix,
            prefix=prefix,
            input_variables=["x", "y_name"],
        )

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

        print("emebdding check5")
        train_x = torch.tensor(embedding_isomap)
        print("emebdding check6")

        train_y = torch.tensor(list(map(float, y))).unsqueeze(-1).double()
        print("train check")
        self.regressor = SingleTaskGP(train_x, train_y)
        print("train check2")
        mll = ExactMarginalLogLikelihood(self.regressor.likelihood, self.regressor)
        fit_gpytorch_mll_torch(mll)

    def _tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> Dict:
        """Tell the optimizer about a new example."""
        if self.use_quantiles:
            self.qt = QuantileTransformer(
                values=self._ys + [y], n_quantiles=self.n_quantiles
            )
            y = self.qt.to_quantiles(y)

        if alt_ys is not None:
            raise ValueError("Alt ys not supported for GPR.")
        example_dict = dict(
            x=self.format_x(x),
            y=self.format_y(y),
            y_name=self._y_name,
        )
        self._ys.append(y)
        inv_dict = dict(
            x=self.format_x(x),
            y=self.format_y(y),
            y_name=self._y_name,
            x_name=self._x_name,
        )
        return example_dict, inv_dict

    def tell(
        self, x: str, y: float, alt_ys: Optional[List[float]] = None, train=True
    ) -> None:
        # Reimplement tell to avoid feeding new points to the prompt exemple_selector
        """Tell the optimizer about a new example."""
        example_dict, inv_example = self._tell(x, y, alt_ys)

        if not self._ready:
            self.prompt = self._setup_prompt(
                None, self._prompt_template, self._suffix, self._prefix
            )
            self._ready = True

        self.examples.append(example_dict)
        self._example_count += 1

        if train:
            try:
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

            except ValueError as e:
                msg = (
                    f"{40*'-'} ERROR {40*'-'}\n"
                    f"{e}\n"
                    f"Not enough data to train.\n"
                    f"We use an isomap considering 5 neighbors. Therefore, more than 6 points are needed to train the model.\n"
                    f"Use train=False to tell N-1 points to the model first.\n"
                    f"Then use train=True to tell the last point to train the model.\n"
                    f"Alternatively, use `pool` to pass a boicl.Pool to train the isomap during AskTellGPR construction.\n"
                    f'{85*"-"}'
                )
                raise ValueError(msg)

    def predict(
        self, x: str, system_message: str = None
    ) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        # Reimplement predict to avoid creating llms and the inverse prompt
        """Predict the probability distribution and values for a given x.

        Args:
            x: The x value(s) to predict.
        Returns:
            The probability distribution and values for the given x.

        """
        if not isinstance(x, list):
            x = [x]
        # if not self.regressor:
        #     raise ValueError("Model not trained. Please provide more data.")
        if not self._ready:
            self.prompt = self._setup_prompt(
                None, self._prompt_template, self._suffix, self._prefix
            )
            self._ready = True

        # if self._selector_k is not None:
        #     self.prompt.example_selector.k = min(self._example_count, self._selector_k)

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

    def _ask(
        self,
        possible_x: List[str],
        best: float,
        aq_fxn: Callable,
        k: int,
        system_message: str,
    ) -> Tuple[List[str], List[float], List[float]]:
        results = self.predict(possible_x, system_message=system_message)
        # drop empties
        if type(results) != type([]):
            results = [results]
        results = [r for r in results if len(r) > 0]
        aq_vals = [aq_fxn(r, best) for r in results]
        selected = np.argsort(aq_vals)[::-1][:k]
        means = [r.mean() for r in results]
        stds = [r.std() for r in results]
        print(selected, means, stds)

        return (
            [possible_x[i] for i in selected],
            [aq_vals[i] for i in selected],
            [means[i] for i in selected],
            [stds[i] for i in selected],
        )

    def ask(
        self,
        possible_x: Union[Pool, List[str]],
        aq_fxn: str = "upper_confidence_bound",
        k: int = 1,
        inv_filter: int = None,
        aug_random_filter: int = None,
        lambda_mult: float = 0.5,
        _lambda: float = 0.5,
        system_message: Optional[str] = "",
        inv_system_message: Optional[str] = "",
    ) -> Tuple[List[str], List[float], List[float]]:
        # if inv_filter:
        #     raise ValueError("Inverse filtering not supported for GPR.")

        return super().ask(
            possible_x,
            aq_fxn,
            k,
            inv_filter=0,
            aug_random_filter=aug_random_filter
            if aug_random_filter
            else len(possible_x),
            lambda_mult=lambda_mult,
            _lambda=_lambda,
            system_message=system_message,
            inv_system_message=inv_system_message,
        )
