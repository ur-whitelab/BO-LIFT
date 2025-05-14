import numpy as np
import os
import re
import openai
from langchain_openai import OpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatAnyscale
from langchain_community.callbacks import get_openai_callback

# from langchain.cache import InMemoryCache
import langchain
from dataclasses import dataclass

from langchain.schema import HumanMessage, SystemMessage
from functools import reduce
from typing import Union
import warnings

# langchain.llm_cache = InMemoryCache()


def truncate(s):
    """Truncate to first number"""
    try:
        return re.findall(r"[-+]?\d*\.\d+|\d+", s)[0]
    except IndexError:
        return s


@dataclass
class DiscreteDist:
    values: np.ndarray
    probs: np.ndarray

    def __post_init__(self):
        # make sure np arrays
        self.values = np.array(self.values)
        self.probs = np.array(self.probs)
        uniq_values = np.unique(self.values)
        if len(uniq_values) < len(self.values):
            # need to mergefg
            uniq_probs = np.zeros(len(uniq_values))
            for i, v in enumerate(uniq_values):
                uniq_probs[i] = np.sum(self.probs[self.values == v])
            self.values = uniq_values
            self.probs = uniq_probs

    def sample(self):
        return np.random.choice(self.values, p=self.probs)

    def mean(self):
        return np.sum(self.values * self.probs)

    def mode(self):
        return self.values[np.argmax(self.probs)]

    def std(self):
        return np.sqrt(
            np.sum((self.values - self.mean()) ** 2 * self.probs)
        )  # should be dividing by the bessel correction and sum of the weights too

    def __repr__(self):
        return f"DiscreteDist({self.values}, {self.probs})"

    def __len__(self):
        return len(self.values)


@dataclass
class GaussDist:
    _mean: float
    _std: float

    def sample(self):
        return np.random.normal(self._mean, self._std)

    def mean(self):
        return self._mean

    def mode(self):
        return self._mean

    def std(self):
        return self._std

    def set_std(self, value):
        self._std = value

    def __repr__(self):
        return f"GaussDist({self._mean}, {self._std})"

    def __len__(self):
        return 1


def make_dd(values, probs):
    dd = DiscreteDist(values, probs)
    if len(dd) == 1:
        return GaussDist(dd.mean(), None)

    return dd


def get_llm(
    model_name: str = "gpt-4o",
    temperature: float = 0.7,
    n: int = 5,
    top_p: int = 1,
    best_of: int = 1,
    max_tokens: int = 128,
    logit_bias: dict = {},
    **kwargs,
):
    openai_models = ["davinci-002", "gpt-3.5-turbo-instruct"]
    chatopenai_models = [
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-4-turbo-preview",
        "gpt-3.5-turbo-0125",
        "gpt-4-0125-preview",
        "gpt-4o",
        "gpt-4o-mini",
    ]
    anyscale_models = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ]
    anthropic_models = [
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
    ]

    kwargs = {
        "model_name": model_name,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "best_of": best_of,
        "max_tokens": max_tokens,
        "logit_bias": logit_bias,
        **kwargs,
    }

    if model_name in openai_models:
        return OpenAILLM(**kwargs)
    elif model_name in chatopenai_models:
        return ChatOpenAILLM(**kwargs)
    elif model_name in anthropic_models:
        return AnthropicLLM(**kwargs)
    elif model_name.startswith("openrouter"):
        kwargs["model_name"] = kwargs["model_name"].replace("openrouter/", "")
        return OpenRouterLLM(**kwargs)
    else:
        warnings.warn(
            f"Model {model_name} not explicitly supported. Please choose from {openai_models + chatopenai_models + anyscale_models + anthropic_models}\n\nWe will try to use this model as a ChatOpenAI model."
        )
        return ChatOpenAILLM(**kwargs)
        # raise ValueError(f"Model {model_name} not supported. Please choose from {openai_models + chatopenai_models}")


class LLM:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        n: int = 1,
        top_p: int = 1,
        best_of: int = 1,
        max_tokens: int = 128,
        logit_bias: dict = {},
        use_logprobs: bool = False,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.n = n
        self.top_p = top_p
        self.best_of = best_of
        self.max_tokens = max_tokens
        self.logit_bias = logit_bias
        self.kwargs = kwargs
        self.llm = self.create_llm()
        self.use_logprobs = use_logprobs

    def create_llm(self):
        raise NotImplementedError("Must be implemented in subclasses")

    def parse_response(self, generations):
        raise NotImplementedError("Must be implemented in subclasses")

    def parse_inv_response(self, generations):
        raise NotImplementedError("Must be implemented in subclasses")

    def predict(self, query_list, inv_pred=False, verbose=False, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclasses")


class OpenAILLM(LLM):
    def create_llm(self):
        self.kwargs.update(
            {
                "logprobs": 5,
            }
        )
        return OpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            n=self.n,
            top_p=self.top_p,
            best_of=self.best_of,
            max_tokens=self.max_tokens,
            logit_bias=self.logit_bias,
            logprobs=5,
            # top_logprobs= True,
            # model_kwargs=self.kwargs
        )

    def predict(self, query_list, inv_pred=False, verbose=False, *args, **kwargs):
        if type(query_list) == str:
            query_list = [query_list]

        if "system_message" in kwargs:
            del kwargs["system_message"]
        with get_openai_callback() as cb:
            completion_response = self.llm.generate(query_list, *args, **kwargs)
            token_usage = cb.total_tokens
        if verbose:
            print("-" * 80)
            print(query_list[0])
            print("-" * 80)
            print(completion_response)
            print(query_list[0], completion_response.generations[0][0].text)
            print("-" * 80)

        results = []
        for gens in completion_response.generations:
            if inv_pred:
                results.append(self.parse_inv_response(gens))
            else:
                results.append(self.parse_response(gens))

        return results, token_usage

    def parse_response(self, generations):
        values, logprobs = [], []
        for gen in generations:
            try:
                v = float(truncate(gen.text))
                values.append(v)
            except ValueError:
                continue
            if self.use_logprobs:
                # can do inner sum because there is only one token
                lp = sum(
                    [
                        sum(
                            reduce(
                                lambda a, b: {**a, **b},
                                gen.generation_info["logprobs"]["top_logprobs"],
                            ).values()
                        )
                    ]
                )
                logprobs.append(lp)
            else:
                logprobs.append(np.log(1.0))

        eps = 1e-15
        probs = np.exp(np.array(logprobs))
        probs = probs / np.sum(probs + eps)

        return make_dd(np.array(values), probs)

    def parse_inv_response(self, generations):
        return generations[0].text


class ChatOpenAILLM(LLM):
    def create_llm(self):
        return ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            n=self.n,
            max_tokens=self.max_tokens,
            logprobs=True,
            top_logprobs=5,
        )

    def predict(self, query_list, inv_pred=False, verbose=False, *args, **kwargs):
        if type(query_list) == str:
            query_list = [query_list]

        if "system_message" in kwargs:
            system_message = kwargs["system_message"]
            del kwargs["system_message"]
        else:
            system_message = ""
            warnings.warn(
                "`system_message` not provided. Not clearly specifying the task for the LLM usually decreases its performance considerably. Please provide a system_message for ChatOpenAI models when invoking the `predict` method."
            )

        system_message_prompt = SystemMessage(content=system_message)
        query_list = [
            [system_message_prompt, HumanMessage(content=q)] for q in query_list
        ]

        with get_openai_callback() as cb:
            completion_response = self.llm.generate(query_list, *args, **kwargs)
            token_usage = cb.total_tokens

        if verbose:
            print("-" * 80)
            print(query_list[0])
            print("-" * 80)
            print(query_list[0], completion_response.generations[0][0].text)
            print("-" * 80)

        results = []
        for gens in completion_response.generations:
            if inv_pred:
                results.append(self.parse_inv_response(gens))
            else:
                results.append(self.parse_response(gens))
        return results, token_usage

    def parse_response(self, generations):
        values, logprobs = [], []
        for gen in generations:
            try:
                v = float(truncate(gen.text))
                values.append(v)
            except ValueError:
                continue
            if self.use_logprobs:
                lp = sum(
                    p["logprob"] for p in gen.generation_info["logprobs"]["content"]
                )
                logprobs.append(lp)
            else:
                logprobs.append(np.log(1.0))

        eps = 1e-15

        probs = np.exp(np.array(logprobs))
        probs = probs / np.sum(probs + eps)

        return make_dd(np.array(values), probs)

    def parse_inv_response(self, generations):
        return generations[0].text


class OpenRouterLLM(LLM):
    def create_llm(self):
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        return client

    def predict(self, query_list, inv_pred=False, verbose=False, *args, **kwargs):
        if type(query_list) == str:
            query_list = [query_list]

        if "system_message" in kwargs:
            system_message = kwargs["system_message"]
            del kwargs["system_message"]
        else:
            system_message = ""

        results = []
        token_usage = 0

        for query in query_list:
            generations = []
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ]
            for _ in range(self.n):
                # Some models doesn't support n. So iterating manually
                completions = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                if completions.choices:
                    generations.append(completions.choices[0].message.content)
                    token_usage += completions.usage.total_tokens
                else:
                    generations.append("")

            if verbose:
                print("-" * 80)
                print(query)
                print("-" * 80)
                print(query, generations)
                print("-" * 80)
            if inv_pred:
                results.append(self.parse_inv_response(generations))
            else:
                results.append(self.parse_response(generations))

        return results, token_usage

    def parse_response(self, generations):
        values, logprobs = [], []
        for gen in generations:
            try:
                v = float(truncate(gen))
                values.append(v)
            except ValueError:
                continue
            logprobs.append(np.log(1.0))
        probs = np.exp(np.array(logprobs))
        probs = probs / np.sum(probs)

        return make_dd(np.array(values), probs)

    def parse_inv_response(self, generations):
        return generations[0]


class AnthropicLLM(LLM):
    def create_llm(self):
        import anthropic

        self.kwargs.update(
            {
                # "logprobs": True,
                # "top_logprobs": 5,
                "n": self.n,
                "best_of": self.best_of,
            }
        )

        return ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

    def predict(self, query_list, inv_pred=False, verbose=False, *args, **kwargs):
        if type(query_list) == str:
            query_list = [query_list]
        if "system_message" in kwargs:
            system_message = kwargs["system_message"]
            del kwargs["system_message"]
        else:
            system_message = ""
            warnings.warn(
                "`system_message` not provided. Not clearly specifying the task for the LLM usually decreases its performance considerably. Please provide a system_message for ChatOpenAI models when invoking the `predict` method."
            )

        if self.use_logprobs:
            self.use_logprobs = False
            warnings.warn(
                "Logprobs not supported for Anthropic models. Ignoring `use_logprobs` parameter."
            )

        system_message_prompt = SystemMessage(content=system_message)
        query_list = [
            [[system_message_prompt, HumanMessage(content=q)] for _ in range(self.n)]
            for q in query_list
        ]

        completion_responses = [
            self.llm.generate(q, *args, **kwargs) for q in query_list
        ]
        # completion_responses = self.llm.generate(query_list, *args, **kwargs)

        results = []
        token_usage = 0
        for gens in completion_responses:
            token_usage += sum(
                [
                    gen[0].message.usage_metadata["total_tokens"]
                    for gen in gens.generations
                ]
            )
            if inv_pred:
                results.append(self.parse_inv_response(gens))
            else:
                results.append(self.parse_response(gens.generations))
            # results.append(gens[0].message.content)

        return results, token_usage

    def parse_response(self, generations):
        values, logprobs = [], []
        for gen in generations:
            try:
                v = float(truncate(gen[0].text))
                values.append(v)
            except ValueError:
                continue

            logprobs.append(np.log(1.0))
        probs = np.exp(np.array(logprobs))
        probs = probs / np.sum(probs)

        return make_dd(np.array(values), probs)

    def parse_inv_response(self, generations):
        return generations[0].text


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(".env")

    llm = get_llm(model_name="openrouter/mistralai/mistral-7b-instruct:free")
    p = llm.predict(
        ["Hello. What is 2 + 2?", "Hello. What is 2 + 3?"],
        system_message="You are a robot who can do math. Answer only the number referent to the answer of the mathematical operation. Nothing else.",
    )

    print(p)
