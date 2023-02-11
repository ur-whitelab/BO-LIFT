import numpy as np
from functools import partial
from typing import *
from .llm_model import get_llm, openai_predict
from .aqfxns import (
    probability_of_improvement,
    expected_improvement,
    upper_confidence_bound,
    greedy,
)
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate


class AskTellFewShot:
    def __init__(
        self,
        prompt_template: PromptTemplate,
        suffix: str,
        model: str = "text-ada-001",
        prefix: str = "",
    ) -> None:
        """Initialize Ask-Tell optimizer.

        You can pass formatters that will make your data more compatible with the model. Note that
        y as output form the model must be a float(can be parsed with ``float(y_str)``)

        Args:
            prompt_template: Prompt template that should take x and y.
            suffix: Suffix to add to the prompt(the first half of your prompt).
            model: OpenAI base model to use for training and inference.
            prefix: Prefix to add before all examples.
        """
        self._trained = 0

        # test out prompt
        try:
            prompt_template.format(x="test", y=-1.2)
        except Exception as e:
            raise ValueError(
                "Could not use prompt - make sure {x} and {y} are the template names."
            ) from e
        self.prompt = self._setup_prompt(prompt_template, suffix, prefix)
        self.train_loss = None
        self.llm = get_llm(model_name=model)
        self._ys = []

    def _setup_prompt(
        self, template: PromptTemplate, suffix: str, prefix: str
    ) -> FewShotPromptTemplate:
        return FewShotPromptTemplate(
            examples=[],
            example_prompt=template,
            suffix=suffix,
            prefix=prefix,
            input_variables=["x"],
        )

    def tell(self, x: str, y: float) -> None:
        """Tell the optimizer about a new example."""
        self.prompt.examples.append(dict(x=x, y=f"{y: 0.2f}"))
        self._ys.append(y)

    def predict(self, x: str) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """Predict the probability distribution and values for a given x.

        Args:
            x: The x value(s) to predict.

        Returns:
            The probability distribution and values for the given x.

        """
        if not isinstance(x, list):
            x = [x]
        queries = [self.prompt.format(x=x_i) for x_i in x]
        results = openai_predict(queries, self.llm)

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
        aq_vals = [aq_fxn(*r, best) for r in results]
        selected = np.argsort(aq_vals)[::-1][:k]
        modes = [r[1][np.argmax(r[0])] for r in results]
        return (
            [possible_x[i] for i in selected],
            [aq_vals[i] for i in selected],
            [modes[i] for i in selected],
        )
