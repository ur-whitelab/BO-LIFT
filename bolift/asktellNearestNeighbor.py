import numpy as np

from typing import *
from .asktell import AskTellFewShotTopk
from .llm_model import GaussDist
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector


class AskTellNearestNeighbor(AskTellFewShotTopk):
    def __init__(self, knn=1, **kwargs):
        super().__init__(selector_k=knn, **kwargs)
        self.knn = knn

    def _tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> Dict:
        """Tell the optimizer about a new example."""
        if alt_ys is not None:
            raise ValueError("Alt ys not supported for topk.")
        example_dict = dict(
            x=self.format_x(x),
            y=self.format_y(y),
            y_name=self._y_name,
        )
        self._ys.append(y)
        return example_dict

    def tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> None:
        """Tell the optimizer about a new example."""
        example_dict = self._tell(x, y, alt_ys)
        # we want to have example
        # to initialize prompts, so send it
        if not self._ready:
            self.prompt = self._setup_prompt(
                example_dict, self._prompt_template, self._suffix, self._prefix
            )
            self._ready = True
        else:
            if self._selector_k is not None:
                self.prompt.example_selector.add_example(example_dict)
            else:
                self.prompt.examples.append(example_dict)
        self._example_count += 1

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
        if example is not None:
            prompt_template.format(**example)
            examples = [example]
        else:
            examples = []
        example_selector = None
        if self._selector_k is not None:
            if len(examples) == 0:
                raise ValueError("Cannot do zero-shot with selector")
            example_selector = (
                example_selector
            ) = SemanticSimilarityExampleSelector.from_examples(
                [example],
                OpenAIEmbeddings(),
                Chroma,
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

    def predict(self, x: str) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        if not isinstance(x, list):
            x = [x]
        if not self._ready:
            self.prompt = self._setup_prompt(
                None, self._prompt_template, self._suffix, self._prefix
            )
            self._ready = True

        if self._selector_k is not None:
            self.prompt.example_selector.k = min(self._example_count, self._selector_k)

        selected = [
            self.prompt.example_selector.select_examples({"x": x_i}) for x_i in x
        ]

        predictions = [[float(s["y"]) for s in selected_i] for selected_i in selected]

        results = [GaussDist(np.mean(p), np.std(p)) for p in predictions]

        if len(x) == 1:
            return results[0]
        return results
