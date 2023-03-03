import numpy as np
import pandas as pd
from .pool import Pool
from .asktell import AskTellFewShotTopk
from .llm_model import GaussDist

from typing import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, PairwiseKernel, RBF
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings


openai_models = ["text-embedding-ada-002","text-embedding-divinci-001","text-embedding-curie-001","text-embedding-babbage-001","text-embedding-ada-001"]

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_regressor()

        if model not in models:   # model = Dagobert42/bert-base-uncased-finetuned-material-synthesis
            self._embedding = HuggingFaceEmbeddings()
        else:
            self._embedding = OpenAIEmbeddings()
        self.examples = []
       
         

    def _set_regressor(self):
        cosine_kernel = PairwiseKernel(metric="rbf")
        constant_kernel = ConstantKernel(
            constant_value=1.0, constant_value_bounds="fixed"
        )
        cos_kernel = constant_kernel * cosine_kernel
        self.regressor = GaussianProcessRegressor(
            kernel=RBF(length_scale=1e-3, length_scale_bounds=(1e-10, 1e1)),
            # kernel=cos_kernel,
            n_restarts_optimizer=2,
            normalize_y=True,
        )

    def _predict(self, X):
        embedding = np.array(self._embedding.embed_documents(X, 250))
        results = []
        means, stds = self.regressor.predict(embedding, return_std=True)
        results = [GaussDist(mean, std) for mean, std in zip(means, stds)]
        return results, 0

    def _tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None):
        example_dict = dict(
            x=self.format_x(x),
            y=self.format_y(y),
            y_name=self._y_name,
        )
        self.examples.append(example_dict)

        self._train(
            [ex["x"] for ex in self.examples], [ex["y"] for ex in self.examples]
        )

        self._ys.append(y)
        inv_dict = dict(
            x=self.format_x(x),
            y=self.format_y(y),
            y_name=self._y_name,
        )
        return example_dict, inv_dict

    def _train(self, X, y):
        embedding = np.array(self._embedding.embed_documents(X, 250))
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
