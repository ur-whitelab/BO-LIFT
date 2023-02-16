import numpy as np
import re
from langchain.llms import OpenAI
from langchain.cache import InMemoryCache
import langchain
from dataclasses import dataclass


@dataclass
class DiscreteDist:
    values: np.ndarray
    probs: np.ndarray

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


langchain.llm_cache = InMemoryCache()


def get_llm(
    model_name="text-babbage-001",
    temperature=0.05,
    n=1,
    top_p=1,
    best_of=1,
    max_tokens=128,
    **kwargs,
):
    return OpenAI(
        model_name=model_name,
        temperature=temperature,
        n=n,
        best_of=best_of,
        top_p=top_p,
        model_kwargs=kwargs,
        max_tokens=max_tokens,
    )


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
    return DiscreteDist(np.array([k for k, v in result]), probs)


def truncate(s):
    """Truncate to first number"""
    try:
        return re.findall(r"[-+]?\d*\.\d+|\d+", s)[0]
    except IndexError:
        return s


def remove_overlap(s1, s2, check_l=10):
    """There may be some of s1 in s2. Remove it and return rest of s2"""
    for i in range(check_l, 0, -1):
        if s1[-i:] == s2[:i]:
            return s2[len(s1[-i:]) :]
    return s2


def parse_response_topk(generations):
    values, logprobs = [], []
    for gen in generations:
        try:
            v = float(truncate(gen.text))
            values.append(v)
        except ValueError:
            continue
        # can do inner sum because there is only one token
        lp = sum(
            [
                sum(x.to_dict().values())
                for x in gen.generation_info["logprobs"]["top_logprobs"]
            ]
        )
        logprobs.append(lp)

    probs = np.exp(np.array(logprobs))
    probs = probs / np.sum(probs)
    return DiscreteDist(np.array(values), probs)


def openai_choice_predict(query_list, llm, *args, **kwargs):
    """Predict the output numbers for a given list of queries"""
    completion_response = llm.generate(query_list, *args, **kwargs)
    results = []
    for gen, q in zip(completion_response.generations, query_list):
        results.append(parse_response(gen[0], q, llm))
    return results


def openai_topk_predict(query_list, llm, *args, **kwargs):
    """Predict the output numbers for a given list of queries"""
    completion_response = llm.generate(query_list, *args, **kwargs)
    results = []
    for gens in completion_response.generations:
        results.append(parse_response_topk(gens))
    return results
