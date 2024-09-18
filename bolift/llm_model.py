import numpy as np
import os
import re
import openai
from langchain_openai import OpenAI, ChatOpenAI
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
        return np.sqrt(np.sum((self.values - self.mean()) ** 2 * self.probs))

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
        model_name      : str   = "gpt-3.5-turbo",
        temperature     : float = 0.7,
        n               : int   = 5,
        top_p           : int   = 1,
        best_of         : int   = 1,
        max_tokens      : int   = 128,
        logit_bias      : dict  = {},
        **kwargs
    ):
    openai_models = ["davinci-002", "gpt-3.5-turbo-instruct"]
    chatopenai_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-3.5-turbo-0125", "gpt-4-0125-preview", "gpt-4o", "gpt-4o-mini"]
    anyscale_models = ["meta-llama/Llama-2-7b-chat-hf","meta-llama/Llama-2-13b-chat-hf","meta-llama/Llama-2-70b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
    
    kwargs = {
        "model_name": model_name,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "best_of": best_of,
        "max_tokens": max_tokens,
        "logit_bias": logit_bias,
        **kwargs
    }

    if model_name in openai_models:
        return OpenAILLM(**kwargs)
    elif model_name in chatopenai_models:
        return ChatOpenAILLM(**kwargs)
    elif model_name in anyscale_models:
        return AnyScaleLLM(**kwargs)
    else:
        warnings.warn(f"Model {model_name} not explicitly supported. Please choose from {openai_models + chatopenai_models + anyscale_models}\n\nWe will try to use this model as a ChatOpenAI model.")
        return ChatOpenAILLM(**kwargs)
        # raise ValueError(f"Model {model_name} not supported. Please choose from {openai_models + chatopenai_models}")


class LLM:
    def __init__(self, 
                 model_name     : str = "gpt-3.5-turbo-instruct", 
                 temperature    : float = 0.7, 
                 n              : int = 1, 
                 top_p          : int = 1, 
                 best_of        : int = 1, 
                 max_tokens     : int = 128, 
                 logit_bias     : dict = {},
                 use_logprobs   : bool = False,
                 **kwargs
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
        self.kwargs.update({
            "logprobs": 5,
        })
        return OpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            n=self.n,
            top_p=self.top_p,
            best_of=self.best_of,
            max_tokens=self.max_tokens,
            logit_bias=self.logit_bias,
            model_kwargs=self.kwargs
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
                        sum(reduce(lambda a, b: {**a, **b}, gen.generation_info["logprobs"]["top_logprobs"]).values())
                    ]
                )
                logprobs.append(lp)
            else:
                logprobs.append(np.log(1.0))

        probs = np.exp(np.array(logprobs))
        probs = probs / np.sum(probs)

        return make_dd(np.array(values), probs)

    def parse_inv_response(self, generations):
        return generations[0].text


class ChatOpenAILLM(LLM):
    def create_llm(self):
        # self.kwargs.update({
        #     "logprobs": True,
        #     "top_logprobs": 5
        # })
        return ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            n=self.n,
            max_tokens=self.max_tokens,
            logprobs= True,
            top_logprobs= 5,
            #model_kwargs=self.kwargs,
        )

    def predict(self, query_list, inv_pred=False, verbose=False, *args, **kwargs):
        if type(query_list) == str:
            query_list = [query_list]

        if "system_message" in kwargs:
            system_message = kwargs["system_message"]
            del kwargs["system_message"]
        else:
            system_message = ""
            warnings.warn("`system_message` not provided. Not clearly specifying the task for the LLM usually decreases its performance considerably. Please provide a system_message for ChatOpenAI models when invoking the `predict` method.")

        system_message_prompt = SystemMessage(
            content=system_message
        )
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
                    p['logprob'] for p in  gen.generation_info["logprobs"]["content"]
                )
                logprobs.append(lp)
            else:
                logprobs.append(np.log(1.0))
        
        probs = np.exp(np.array(logprobs))
        probs = probs / np.sum(probs)

        return make_dd(np.array(values), probs)
    
    def parse_inv_response(self, generations):
        return generations[0].text
   

class AnyScaleLLM(ChatOpenAILLM):
    def create_llm(self):
        if "logprobs" in self.kwargs:
            del self.kwargs["logprobs"] # not supported

        return ChatAnyscale(
            model_name=self.model_name,
            temperature=self.temperature,
            n=self.n,
            max_tokens=self.max_tokens,
            model_kwargs=self.kwargs,
        )

# Code below is deprecated and will be removed in future versions

def wrap_chatllm(query_list, llm, system_message=""):
    if type(query_list) == str:
        query_list = [query_list]
    if type(llm) == ChatOpenAI:
        system_message_prompt = SystemMessage(
            content=system_message
        )
        query_list = [
            [system_message_prompt, HumanMessage(content=q)] for q in query_list
        ]
    return query_list


def parse_response(generation, prompt, llm):
    # first parse the options into numbers
    text = generation.text
    matches = re.findall(r"([A-Z])\. .*?([\+\-\d][\d\.e]*)", text)
    values = dict()
    k = None
    for m in matches:
        try:
            k, v = m[0], float(m[1])
            values[k] = v
        except ValueError:
            pass
        k = None
    # now get log prob of tokens after Answer:
    tokens = generation.generation_info["logprobs"]["top_logprobs"]
    offsets = generation.generation_info["logprobs"]["text_offset"]
    if "Answer:" not in text:
        # try to extend
        c_generation = llm.generate([prompt + text + "\nAnswer:"]).generations[0][0]

        logprobs = c_generation.generation_info["logprobs"]["top_logprobs"][0]
    else:
        # find token probs for answer
        # feel like this is supper brittle, but not sure what else to try
        at_answer = False
        for i in range(len(offsets)):
            start = offsets[i] - offsets[0]
            end = offsets[i + 1] - offsets[0] if i < len(offsets) - 1 else -1
            selected_token = text[start:end]
            if "Answer" in selected_token:
                at_answer = True
            if at_answer and selected_token.strip() in values:
                break
        logprobs = tokens[i]
    result = [
        (values[k.strip()], v) for k, v in logprobs.items() if k.strip() in values
    ]
    probs = np.exp(np.array([v for k, v in result]))
    probs = probs / np.sum(probs)
    # return DiscreteDist(np.array([k for k, v in result]), probs)
    return make_dd(np.array([k for k, v in result]), probs)


def openai_choice_predict(query_list, llm, verbose, *args, **kwargs):
    """Predict the output numbers for a given list of queries"""
    with get_openai_callback() as cb:
        query_list = wrap_chatllm(query_list, llm)
        completion_response = llm.generate(query_list, *args, **kwargs)
        token_usage = cb.total_tokens
    if verbose:
        print("-" * 80)
        print(query_list[0])
        print("-" * 80)
        print(query_list[0], completion_response.generations[0][0].text)
        print("-" * 80)
    results = []
    for gen, q in zip(completion_response.generations, query_list):
        results.append(parse_response(gen[0], q, llm))
    return results, token_usage

