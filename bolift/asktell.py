import numpy as np
from functools import partial
from typing import *
from .llm_model import get_llm, openai_choice_predict, openai_topk_predict, DiscreteDist
from .aqfxns import (
    probability_of_improvement,
    expected_improvement,
    upper_confidence_bound,
    greedy,
)
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

_answer_choices = ["A", "B", "C", "D", "E"]


class AskTellFewShotMulti:
    def __init__(
        self,
        prompt_template: PromptTemplate = None,
        suffix: Optional[str] = None,
        model: str = "text-curie-001",
        temperature: Optional[float] = None,
        prefix: Optional[str] = None,
        x_formatter: Callable[[str], str] = lambda x: x,
        y_formatter: Callable[[float], str] = lambda y: f"{y: 0.2f}",
        y_name: str = "y",
        selector_k: Optional[int] = None,
    ) -> None:
        """Initialize Ask-Tell optimizer.

        You can pass formatters that will make your data more compatible with the model. Note that
        y as output form the model must be a float(can be parsed with ``float(y_str)``)

        Args:
            prompt_template: Prompt template that should take x and y (for few shot templates)
            suffix: Matching suffix for first part of prompt template - for actual completion.
            model: OpenAI base model to use for training and inference.
            temperature: Temperature to use for inference. If None, will use model default.
            prefix: Prefix to add before all examples (e.g., some context for the model).
            x_formatter: Function to format x for prompting.
            y_formatter: Function to format y for prompting.
            y_name: Name of y variable in prompt template (e.g., density, value of function, etc.)
            selector_k: What k to use when switching to selection mode. If None, will use all examples
        """
        self._selector_k = selector_k
        self._ready = False
        self._ys = []
        self._x_formatter = x_formatter
        self._y_formatter = y_formatter
        self._y_name = y_name
        self._prompt_template = prompt_template
        self._suffix = suffix
        self._prefix = prefix
        self._model = model
        self._example_count = 0

    def _setup_llm(self, model: str, temperature: Optional[float] = None):
        return get_llm(
            model_name=model,
            # put stop with suffix, so it doesn't start babbling
            stop=[self.prompt.suffix.split()[0]],
            max_tokens=256,
            logprobs=5,
            temperature=0.05 if temperature is None else temperature,
        )

    def _setup_prompt(
        self,
        example: Dict,
        prompt_template: Optional[PromptTemplate] = None,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> FewShotPromptTemplate:
        if prefix is None:
            prefix = "For each question, select the correct answer from the five choices below. Indicate your choice with 'Answer: ':\n"
        if prompt_template is None:
            prompt_template = PromptTemplate(
                input_variables=["x", "Answer", "i", "y_name"] + _answer_choices,
                template="Question {i}: Given {x}. What is {y_name}?\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\n\nAnswer: {Answer}\n\n",
            )
            if suffix is not None:
                raise ValueError(
                    "Cannot provide suffix if using default prompt template."
                )
            suffix = "Question {i}: Given {x}, what is {y_name}?\nA."
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
            example_selector = (
                example_selector
            ) = SemanticSimilarityExampleSelector.from_examples(
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
            input_variables=["x", "i", "y_name"],
        )

    def _tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> Dict:
        # implementation of tell
        if alt_ys is not None:
            if len(alt_ys) != len(_answer_choices) - 1:
                raise ValueError("Must provide 4 alternative ys.")
            alt_ys = [self._y_formatter(alt_y) for alt_y in alt_ys]
        else:
            alt_ys = []
            alt_y = y
            for i in range(100):
                if len(alt_ys) == len(_answer_choices) - 1:
                    break
                if i < 50:
                    alt_y = y * 10 ** np.random.normal(0, 0.2)
                else:  # try something different
                    alt_y = y + np.random.uniform(-10, 10)
                if self._y_formatter(alt_y) not in alt_ys and self._y_formatter(
                    alt_y
                ) != self._y_formatter(y):
                    alt_ys.append(self._y_formatter(alt_y))
        # choose answer
        answer = np.random.choice(_answer_choices)
        example_dict = dict(
            x=self._x_formatter(x),
            i=str(self._example_count + 1),
            Answer=answer,
            y_name=self._y_name,
        )
        for a in _answer_choices:
            if a == answer:
                example_dict[a] = self._y_formatter(y)
            else:
                example_dict[a] = alt_ys.pop()
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
            self.llm = self._setup_llm(self._model)
            self._ready = True
        else:
            # in else, so we don't add twice
            if self._selector_k is not None:
                self.prompt.example_selector.add_example(example_dict)
            else:
                self.prompt.examples.append(example_dict)
        self._example_count += 1

    def _predict(self, queries: List[str]) -> List[DiscreteDist]:
        return openai_choice_predict(queries, self.llm)

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
            self.llm = self._setup_llm(self._model)
            self._ready = True

        queries = [
            self.prompt.format(
                x=self._x_formatter(x_i),
                i=str(self._example_count + 1),
                y_name=self._y_name,
            )
            for x_i in x
        ]
        results = self._predict(queries)

        # compute mean and standard deviation
        if len(x) == 1:
            return results[0]
        return results

    def ask(
        self,
        possible_x: List[str],
        aq_fxn: str = "upper_confidence_bound",
        k: int = 1,
        _lambda: float = 0.5,
    ) -> Tuple[List[str], List[float], List[float]]:
        """Ask the optimizer for the next x to try.



        Args:
            possible_x: List of possible x values to choose from.
            aq_fxn: Acquisition function to use.
            k: Number of x values to return.
            _lambda: Lambda value to use for UCB
        Return:
            The selected x values, their acquisition function values, and the predicted y modes.
            Sorted by acquisition function value (descending)
        """
        if aq_fxn == "probability_of_improvement":
            aq_fxn = probability_of_improvement
        elif aq_fxn == "expected_improvement":
            aq_fxn = expected_improvement
        elif aq_fxn == "upper_confidence_bound":
            aq_fxn = partial(upper_confidence_bound, _lambda=_lambda)
        elif aq_fxn == "greedy":
            aq_fxn = greedy
        else:
            raise ValueError(f"Unknown acquisition function: {aq_fxn}")

        if len(self._ys) == 0:
            best = 0
        else:
            best = np.max(self._ys)
        results = self.predict(possible_x)
        # drop empties
        results = [r for r in results if len(r.probs) > 0]
        aq_vals = [aq_fxn(r.probs, r.values, best) for r in results]
        selected = np.argsort(aq_vals)[::-1][:k]
        means = [r.mean() for r in results]
        return (
            [possible_x[i] for i in selected],
            [aq_vals[i] for i in selected],
            [means[i] for i in selected],
        )


class AskTellFewShotTopk(AskTellFewShotMulti):
    def _predict(self, queries: List[str]) -> List[DiscreteDist]:
        return openai_topk_predict(queries, self.llm)

    def _setup_llm(self, model: str, temperature: Optional[float] = None):
        # nucleus sampling seems to get more diversity
        return get_llm(
            n=5,
            best_of=5,
            temperature=1 if temperature is None else temperature,
            model_name=model,
            top_p=0.95,
            stop=["\n", "###", "#", "##"],
            logit_bias={
                "198": -100,  # new line,
                "628": -100,  # double new line,
                "50256": -100,  # endoftext
            },
            max_tokens=32,
            logprobs=1,
        )

    def _setup_prompt(
        self,
        example: Dict,
        prompt_template: Optional[PromptTemplate] = None,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> FewShotPromptTemplate:
        if prefix is None:
            prefix = (
                "Answer correctly the following questions.\n"
                "Your answers must be numerical and "
                "you should end your answer with ###\n\n"
            )
        if prompt_template is None:
            prompt_template = PromptTemplate(
                input_variables=["x", "y", "i", "y_name"],
                template="Q{i}: Given {x}, what is {y_name}?\nA{i}: {y}###\n\n",
            )
            if suffix is not None:
                raise ValueError(
                    "Cannot provide suffix if using default prompt template."
                )
            suffix = "Q{i}: Given {x}. What is {y_name}?\nA{i}: "
        elif suffix is None:
            raise ValueError("Must provide suffix if using custom prompt template.")
        # test out prompt
        if example is not None:
            prompt_template.format(**example)
            examples = [example]
        # TODO: make fake example text
        else:
            examples = []
        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=prompt_template,
            suffix=suffix,
            prefix=prefix,
            input_variables=["x", "i", "y_name"],
        )

    def _tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> Dict:
        """Tell the optimizer about a new example."""
        if alt_ys is not None:
            raise ValueError("Alt ys not supported for topk.")
        example_dict = dict(
            x=self._x_formatter(x),
            i=self._example_count + 1,
            y=self._y_formatter(y),
            y_name=self._y_name,
        )
        self._ys.append(y)
        return example_dict

class AskTellFewShotDirect(AskTellFewShotMulti):
    def _predict(self, queries: List[str]) -> List[DiscreteDist]:
        return openai_topk_predict(queries, self.llm)

    def _setup_llm(self, model: str, temperature: Optional[float] = None):
        return get_llm(
            n=1,
            temperature=0.05 if temperature is None else temperature,
            model_name=model,
            stop=["\n", "###", "#", "##"],
            logit_bias={
                "198": -100,  # new line,
                "628": -100,  # double new line,
                "50256": -100,  # endoftext
            },
            max_tokens=32,
            logprobs=1,
        )

    def _setup_prompt(
        self,
        example: Dict,
        prompt_template: Optional[PromptTemplate] = None,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> FewShotPromptTemplate:
        if prefix is None:
            prefix = (
                "Answer correctly the following questions.\n"
                "Your should answer with a number and uncertainty (1 std dev).  "
                "Here is an example: 3.4 +/- 0.2###\n\n"
            )
        if prompt_template is None:
            prompt_template = PromptTemplate(
                input_variables=["x", "y", "dy", "i", "y_name"],
                template="Q{i}: Given {x}, what is {y_name}?\nA{i}: {y} +/- {dy}###\n\n",
            )
            if suffix is not None:
                raise ValueError(
                    "Cannot provide suffix if using default prompt template."
                )
            suffix = "Q{i}: Given {x}. What is {y_name}?\nA{i}: "
        elif suffix is None:
            raise ValueError("Must provide suffix if using custom prompt template.")
        # test out prompt
        if example is not None:
            prompt_template.format(**example)
            examples = [example]
        # TODO: make fake example text
        else:
            examples = []
        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=prompt_template,
            suffix=suffix,
            prefix=prefix,
            input_variables=["x", "i", "y_name"],
        )

    def _tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> Dict:
        """Tell the optimizer about a new example."""
        if alt_ys is not None:
            raise ValueError("Alt ys not supported for topk.")
        example_dict = dict(
            x=self._x_formatter(x),
            i=self._example_count + 1,
            y=self._y_formatter(y),
            y_name=self._y_name,
        )
        self._ys.append(y)
        return example_dict
