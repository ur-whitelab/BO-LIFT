import numpy as np
from functools import partial
from typing import *
from .llm_model import (
    get_llm,
    openai_choice_predict,
    openai_topk_predict,
    DiscreteDist,
    GaussDist,
    wrap_chatllm,
)
from .aqfxns import (
    probability_of_improvement,
    expected_improvement,
    upper_confidence_bound,
    greedy,
)
from .pool import Pool
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)

_answer_choices = ["A", "B", "C", "D", "E"]


class QuantileTransformer:
    def __init__(self, values, n_quantiles):
        self.n_quantiles = n_quantiles
        self.quantiles = np.linspace(0, 1, n_quantiles + 1)
        self.values_quantiles = np.quantile(values, self.quantiles)

    def to_quantiles(self, values):
        quantile_scores = np.digitize(values, self.values_quantiles[1:-1])
        return quantile_scores

    def to_values(self, quantile_scores):
        values_from_scores = np.interp(
            quantile_scores, range(self.n_quantiles + 1), self.values_quantiles
        )
        return values_from_scores


class AskTellFewShotMulti:
    def __init__(
        self,
        prompt_template: PromptTemplate = None,
        suffix: Optional[str] = None,
        model: str = "text-curie-001",
        temperature: Optional[float] = None,
        prefix: Optional[str] = None,
        x_formatter: Callable[[str], str] = lambda x: x,
        y_formatter: Callable[[float], str] = lambda y: f"{y:0.2f}",
        y_name: str = "output",
        x_name: str = "input",
        selector_k: Optional[int] = None,
        k: int = 5,
        use_quantiles: bool = False,
        n_quantiles: int = 100,
        verbose: bool = False,
        cos_sim: bool = False,
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
            x_name: Name of x variable in prompt template (e.g., input, x, etc.). Only appears in inverse prompt
            selector_k: What k to use when switching to selection mode. If None, will use all examples
            k: Number of examples to use for each prediction.
            verbose: Whether to print out debug information.
        """
        self._selector_k = selector_k
        self._ready = False
        self._ys = []
        self.format_x = x_formatter
        self.format_y = y_formatter
        self._y_name = y_name
        self._x_name = x_name
        self._prompt_template = prompt_template
        self._suffix = suffix
        self._prefix = prefix
        self._model = model
        self._example_count = 0
        self._temperature = temperature
        self._k = k
        self._answer_choices = _answer_choices[:k]
        self.use_quantiles = use_quantiles
        self.n_quantiles = n_quantiles
        self._calibration_factor = None
        self._verbose = verbose
        self.tokens_used = 0
        self.cos_sim = cos_sim

    def _setup_llm(self, model: str, temperature: Optional[float] = None):
        return get_llm(
            model_name=model,
            # put stop with suffix, so it doesn't start babbling
            stop=[self.prompt.suffix.split()[0], self.inv_prompt.suffix.split()[0]],
            max_tokens=256,
            logprobs=self._k,
            temperature=0.05 if temperature is None else temperature,
        )

    def _setup_inv_llm(self, model: str, temperature: Optional[float] = None):
        return get_llm(
            model_name=model,
            # put stop with suffix, so it doesn't start babbling
            stop=[
                self.prompt.suffix.split()[0],
                self.inv_prompt.suffix.split()[0],
                # "\n",
            ],
            max_tokens=256,
            temperature=0.05 if temperature is None else temperature,
        )

    def _setup_inverse_prompt(self, example: Dict):
        prompt_template = PromptTemplate(
            input_variables=["x", "y", "y_name", "x_name"],
            template="If {y_name} is {y}, then {x_name} is {x}\n\n",
        )
        if example is not None:
            prompt_template.format(**example)
            examples = [example]
        else:
            examples = []
        example_selector = None
        if self._selector_k is not None:
            if len(examples) == 0:
                raise ValueError("Cannot do zero-shot with selector")
            if not self.cos_sim:
                example_selector = (
                    example_selector
                ) = MaxMarginalRelevanceExampleSelector.from_examples(
                    [example],
                    OpenAIEmbeddings(),
                    FAISS,
                    k=self._selector_k,
                )
            else:
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
            suffix="If {y_name} is {y}, then {x_name} is ",
            input_variables=["y", "y_name", "x_name"],
        )

    def _setup_prompt(
        self,
        example: Dict,
        prompt_template: Optional[PromptTemplate] = None,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> FewShotPromptTemplate:
        if prefix is None:
            prefix = f"For each multiple choice question, select correct choice from {','.join(self._answer_choices)}\n"
        if prompt_template is None:
            prompt_template = PromptTemplate(
                input_variables=["x", "Answer", "y_name"] + self._answer_choices,
                template="Q: Given {x}. What is {y_name}?\n"
                + "\n".join([f"{a}. {{{a}}}" for a in self._answer_choices])
                + "\nAnswer: {Answer}\n\n",
            )
            if suffix is not None:
                raise ValueError(
                    "Cannot provide suffix if using default prompt template."
                )
            suffix = "Q: Given {x}, what is {y_name}?\nA."
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
            if not self.cos_sim:
                example_selector = (
                    example_selector
                ) = MaxMarginalRelevanceExampleSelector.from_examples(
                    [example],
                    OpenAIEmbeddings(),
                    FAISS,
                    k=self._selector_k,
                )
            else:
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

    def _tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> Dict:
        # implementation of tell
        if alt_ys is not None:
            if len(alt_ys) != len(self._answer_choices) - 1:
                raise ValueError("Must provide 4 alternative ys.")
            alt_ys = [self.format_y(alt_y) for alt_y in alt_ys]
        else:
            alt_ys = []
            alt_y = y
            for i in range(100):
                if len(alt_ys) == len(self._answer_choices) - 1:
                    break
                if i < 50:
                    alt_y = y * 10 ** np.random.normal(0, 0.2)
                else:  # try something different
                    alt_y = y + np.random.uniform(-10, 10)
                if self.format_y(alt_y) not in alt_ys and self.format_y(
                    alt_y
                ) != self.format_y(y):
                    alt_ys.append(self.format_y(alt_y))
        # choose answer
        answer = np.random.choice(self._answer_choices)
        example_dict = dict(
            x=self.format_x(x),
            Answer=answer,
            y_name=self._y_name,
        )
        for a in self._answer_choices:
            if a == answer:
                example_dict[a] = self.format_y(y)
            else:
                example_dict[a] = alt_ys.pop()
        self._ys.append(y)
        inv_example = dict(
            x=self.format_x(x),
            y_name=self._y_name,
            y=self.format_y(y),
            x_name=self._x_name,
        )
        return example_dict, inv_example

    def tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> None:
        """Tell the optimizer about a new example."""
        example_dict, inv_example = self._tell(x, y, alt_ys)
        # we want to have example
        # to initialize prompts, so send it
        if not self._ready:
            self.prompt = self._setup_prompt(
                example_dict, self._prompt_template, self._suffix, self._prefix
            )
            self.inv_prompt = self._setup_inverse_prompt(inv_example)
            self.llm = self._setup_llm(self._model, self._temperature)
            self.inv_llm = self._setup_inv_llm(self._model, self._temperature)
            self._ready = True
        else:
            # in else, so we don't add twice
            if self._selector_k is not None:
                self.prompt.example_selector.add_example(example_dict)
                self.inv_prompt.example_selector.add_example(inv_example)
            else:
                self.prompt.examples.append(example_dict)
                self.inv_prompt.examples.append(inv_example)
        self._example_count += 1

    def _predict(self, queries: List[str]) -> List[DiscreteDist]:
        return openai_choice_predict(queries, self.llm, self._verbose)

    def inv_predict(self, y: float) -> str:
        """A rough inverse model"""
        if not self._ready:
            raise ValueError(
                "Must tell at least one example before inverse predicting."
            )
        query = self.inv_prompt.format(
            y=self.format_y(y), y_name=self._y_name, x_name=self._x_name
        )
        query = wrap_chatllm(query, self.inv_llm)
        x = self.inv_llm(query)
        if type(x) != str:
            return x.content
        return x

    def set_calibration_factor(self, calibration_factor):
        self._calibration_factor = calibration_factor

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

        # need to replace any GaussDist with pop std
        for i, result in enumerate(results):
            if len(self._ys) > 1:
                ystd = np.std(self._ys)
            elif len(self._ys) == 1:
                ystd = self._ys[0]
            else:
                ystd = 10
            if isinstance(result, GaussDist):
                results[i].set_std(ystd)

        if self._calibration_factor:
            for i, result in enumerate(results):
                if isinstance(result, GaussDist):
                    results[i].set_std(result.std() * self._calibration_factor)
                elif isinstance(result, DiscreteDist):
                    results[i] = GaussDist(
                        results[i].mean(),
                        results[i].std() * self._calibration_factor,
                    )

        # compute mean and standard deviation
        if len(x) == 1:
            return results[0]
        return results

    def ask(
        self,
        possible_x: Union[Pool, List[str]],
        aq_fxn: str = "upper_confidence_bound",
        k: int = 1,
        inv_filter: int = 16,
        _lambda: float = 0.5,
    ) -> Tuple[List[str], List[float], List[float]]:
        """Ask the optimizer for the next x to try.



        Args:
            possible_x: List of possible x values to choose from.
            aq_fxn: Acquisition function to use.
            k: Number of x values to return.
            inv_filter: Reduce pool size to this number with inverse model. If 0, not used
            _lambda: Lambda value to use for UCB
        Return:
            The selected x values, their acquisition function values, and the predicted y modes.
            Sorted by acquisition function value (descending)
        """
        if type(possible_x) == type([]):
            possible_x = Pool(possible_x, self.format_x)

        if aq_fxn == "probability_of_improvement":
            aq_fxn = probability_of_improvement
        elif aq_fxn == "expected_improvement":
            aq_fxn = expected_improvement
        elif aq_fxn == "upper_confidence_bound":
            aq_fxn = partial(upper_confidence_bound, _lambda=_lambda)
        elif aq_fxn == "greedy":
            aq_fxn = greedy
        elif aq_fxn == "random":
            return (
                possible_x.sample(k),
                [0] * k,
                [0] * k,
            )
        else:
            raise ValueError(f"Unknown acquisition function: {aq_fxn}")

        if len(self._ys) == 0:
            best = 0
        else:
            best = np.max(self._ys)
        if inv_filter != 0 and inv_filter < len(possible_x):
            # not sure if this is too important,
            # but to get some diversity we
            # take last added y AND best
            approx_x = self.inv_predict(best * np.random.normal(1.0, 0.05))
            possible_x_l = possible_x.approx_sample(approx_x, inv_filter)

        else:
            possible_x_l = list(possible_x)
        results = self._ask(possible_x_l, best, aq_fxn, k)
        if len(results[0]) == 0 and len(possible_x_l) != 0:
            # if we have nothing, just return random one
            return (
                possible_x.sample(k),
                [0] * k,
                [0] * k,
            )
        return results

    def _ask(
        self, possible_x: List[str], best: float, aq_fxn: Callable, k: int
    ) -> Tuple[List[str], List[float], List[float]]:
        results = self.predict(possible_x)
        # drop empties
        results = [r for r in results if len(r) > 0]
        aq_vals = [aq_fxn(r, best) for r in results]
        selected = np.argsort(aq_vals)[::-1][:k]
        means = [r.mean() for r in results]
        return (
            [possible_x[i] for i in selected],
            [aq_vals[i] for i in selected],
            [means[i] for i in selected],
        )


class AskTellFewShotTopk(AskTellFewShotMulti):
    def _predict(self, queries: List[str]) -> List[DiscreteDist]:
        result, token_usage = openai_topk_predict(queries, self.llm, self._verbose)
        if self.use_quantiles and self.qt is None:
            raise ValueError(
                "Can't use quantiles without building the quantile transformer"
            )
        if self.use_quantiles:
            for r in result:
                if isinstance(r, GaussDist):
                    r._mean = self.qt.to_values(r._mean)
                elif isinstance(r, DiscreteDist):
                    r.values = self.qt.to_values(r.values)
        return result, token_usage

    def _setup_llm(self, model: str, temperature: Optional[float] = None):
        # nucleus sampling seems to get more diversity
        return get_llm(
            n=self._k,
            best_of=self._k,
            temperature=0.1 if temperature is None else temperature,
            model_name=model,
            top_p=1.0,
            stop=["\n", "###", "#", "##"],
            logit_bias={
                "198": -100,  # new line,
                "628": -100,  # double new line,
                "50256": -100,  # endoftext
            },
            max_tokens=256,
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
            if not self.cos_sim:
                example_selector = (
                    example_selector
                ) = MaxMarginalRelevanceExampleSelector.from_examples(
                    [example],
                    OpenAIEmbeddings(),
                    FAISS,
                    k=self._selector_k,
                )
            else:
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

    def _tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> Dict:
        """Tell the optimizer about a new example."""

        if self.use_quantiles:
            self.qt = QuantileTransformer(
                values=self._ys + [y], n_quantiles=self.n_quantiles
            )
            y = self.qt.to_quantiles(y)

        if alt_ys is not None:
            raise ValueError("Alt ys not supported for topk.")
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
