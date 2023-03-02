import openai
import numpy as np
import pandas as pd
from .pool import Pool
from .asktell import AskTellFewShotMulti
from .llm_model import DiscreteDist
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

from typing import *
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, PairwiseKernel
# from modAL.models import BayesianOptimizer
# from modAL.acquisition import optimizer_EI, max_EI

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

class AskTellGPR(AskTellFewShotMulti):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_regressor()
        self.examples = []
   

    def _set_regressor(self):
        cosine_kernel = PairwiseKernel(metric=cosine_similarity)
        constant_kernel = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed")
        cos_kernel = constant_kernel * cosine_kernel
        self.regressor = GaussianProcessRegressor(
                                    kernel=cos_kernel,
                                    n_restarts_optimizer=100,
                                    alpha=0.001,
                                    normalize_y=False)



    def get_embedding(self, text, model):
        response = openai.Embedding.create(input = text, model=model)['data']
        return [
            response[emb]['embedding']
            for emb in range(len(response))
            ]


    def _predict(self, X):
        X = X if isinstance(X, list) else [X]
        embedding = np.array(self.get_embedding(X, "text-embedding-ada-002"))
        results=[]
        for emb in embedding:
            pred = self.regressor.predict(embedding, return_std=True)
            results.append(DiscreteDist(pred[0], pred[1]))
        return results, embedding.shape[0]*embedding.shape[-1]
    

    def _train(self, X, y):
        embedding = np.array(self.get_embedding(X, "text-embedding-ada-002"))
        self.regressor.fit(embedding, list(map(float, y)))


    def _tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None):
        example_dict = dict(
            x=self.format_x(x),
            y=self.format_y(y),
            y_name=self._y_name,
        )
        self.examples.append(example_dict)

        self._train(
            [ex['x'] for ex in self.examples],
            [ex['y'] for ex in self.examples]
            )

        self._ys.append(y)
        inv_dict = dict(
            x=self.format_x(x),
            y=self.format_y(y),
            y_name=self._y_name,
        )
        return example_dict, inv_dict

    def _setup_prompt(
            self,
            example: Dict,
            prompt_template: Optional[PromptTemplate] = None,
            suffix: Optional[str] = None,
            prefix: Optional[str] = None,
        ) -> FewShotPromptTemplate:
        if self._selector_k is not None:
            raise ValueError("Cannot use example selector with GPR")
        if prefix is None:
            prefix = (
                "I am a machine learning model that correctly answer questions.\n"
                "Each answer is numeric and ends with ###\n"
            )
        if prompt_template is None:
            prompt_template = PromptTemplate(
                input_variables=["x", "y", "y_name"],
                template="Q: What is {y_name} of {x}?\nA: {y}###\n\n",
            )
        if suffix is None:
            suffix = "Q: What is {y_name} of {x}?\nA: "
        if example is not None:
            examples = [example]
        else:
            examples = []
        example_selector = None
        
        return FewShotPromptTemplate(
            examples=examples if example_selector is None else None,
            example_prompt=prompt_template,
            example_selector=example_selector,
            suffix=suffix,
            prefix=prefix,
            input_variables=["x", "y_name"],
        )
