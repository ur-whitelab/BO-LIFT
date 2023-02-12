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

_answer_choices = ["A", "B", "C", "D", "E"]


class AskTellFewShot:
    def __init__(
        self,
        prompt_template: PromptTemplate = None,
        suffix: Optional[str] = None,
        model: str = "text-curie-001",
        prefix: Optional[str] = None,
        x_formatter: Callable[[str], str] = lambda x: x,
        y_formatter: Callable[[float], str] = lambda y: f"{y: 0.2f}",
        y_name: str = "y",
    ) -> None:
        """Initialize Ask-Tell optimizer.

        You can pass formatters that will make your data more compatible with the model. Note that
        y as output form the model must be a float(can be parsed with ``float(y_str)``)

        Args:
            prompt_template: Prompt template that should take x and y (for few shot templates)
            suffix: Matching suffix for first part of prompt template - for actual completion.
            model: OpenAI base model to use for training and inference.
            prefix: Prefix to add before all examples (e.g., some context for the model).
            x_formatter: Function to format x for prompting.
            y_formatter: Function to format y for prompting.
            y_name: Name of y variable in prompt template (e.g., density, value of function, etc.)
        """
        self._trained = 0

        self.prompt = self._setup_prompt(prompt_template, suffix, prefix)
        self.train_loss = None
        self.llm = self._setup_llm(model)
        self._ys = []
        self._x_formatter = x_formatter
        self._y_formatter = y_formatter
        self._y_name = y_name

    def _setup_llm(self, model: str):
        return get_llm(
            model_name=model,
            # put stop with suffix, so it doesn't start babbling
            stop=[self.prompt.suffix.split()[0]],
            max_tokens=128,
            logprobs=5,
        )

    def _setup_prompt(
        self,
        prompt_template: Optional[PromptTemplate] = None,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> FewShotPromptTemplate:
        if prefix is None:
            prefix = "For each question, select the best answer from the five choices below:\n"
        if prompt_template is None:
            prompt_template = PromptTemplate(
                input_variables=["x", "Answer", "i", "y_name"] + _answer_choices,
                template="Question {i}: Given {x}, what is {y_name}?\n A. {A}\n B. {B}\n C. {C}\n D. {D}\n E. {E}\n\nAnswer: {Answer}\n\n",
            )
            if suffix is not None:
                raise ValueError(
                    "Cannot provide suffix if using default prompt template."
                )
            suffix = "Question {i}: Given {x}, what is {y_name}?\n A."
        elif suffix is None:
            raise ValueError("Must provide suffix if using custom prompt template.")
        # test out prompt
        prompt_template.format(
            x="test",
            A="12",
            B="32",
            C="20",
            D="16",
            E="24",
            Answer="C",
            i=1,
            y_name="fe",
        )
        return FewShotPromptTemplate(
            examples=[],
            example_prompt=prompt_template,
            suffix=suffix,
            prefix=prefix,
            input_variables=["x", "i", "y_name"],
        )

    def tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> None:
        """Tell the optimizer about a new example."""
        if alt_ys is not None:
            if len(alt_ys) != 4:
                raise ValueError("Must provide 4 alternative ys.")
        else:
            alt_ys = [y * n for n in np.random.uniform(0.8, 1.2, 4)]
        # choose answer
        answer = np.random.choice(_answer_choices)
        example_dict = dict(
            x=self._x_formatter(x),
            i=len(self.prompt.examples) + 1,
            Answer=answer,
            y_name=self._y_name,
        )
        for a in _answer_choices:
            if a == answer:
                example_dict[a] = self._y_formatter(y)
            else:
                example_dict[a] = self._y_formatter(alt_ys.pop())
        self.prompt.examples.append(example_dict)
        self._ys.append(y)

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
        queries = [
            self.prompt.format(
                x=self._x_formatter(x_i),
                i=len(self.prompt.examples) + 1,
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
        modes = [r.mode() for r in results]
        return (
            [possible_x[i] for i in selected],
            [aq_vals[i] for i in selected],
            [modes[i] for i in selected],
        )


class AskTellFewShotTopk(AskTellFewShot):
    def _predict(self, queries: List[str]) -> List[DiscreteDist]:
        return openai_topk_predict(queries, self.llm)

    def _setup_llm(self, model: str):
        # nucleus sampling seems to get more diversity
        return get_llm(
            n=5,
            best_of=5,
            temperature=1,
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
            suffix = "Q{i}: Given {x}, what is {y_name}?\nA{i}: "
        elif suffix is None:
            raise ValueError("Must provide suffix if using custom prompt template.")
        # test out prompt
        prompt_template.format(x="test", y="2.3", i=1, y_name="tee")
        return FewShotPromptTemplate(
            examples=[],
            example_prompt=prompt_template,
            suffix=suffix,
            prefix=prefix,
            input_variables=["x", "i", "y_name"],
        )

    def tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> None:
        """Tell the optimizer about a new example."""
        if alt_ys is not None:
            raise ValueError("Alt ys not supported for topk.")
        self.prompt.examples.append(
            dict(
                x=self._x_formatter(x),
                i=len(self.prompt.examples) + 1,
                y=self._y_formatter(y),
                y_name=self._y_name,
            )
        )
        self._ys.append(y)
