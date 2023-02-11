import numpy as np
import re
from langchain.llms import OpenAI
from langchain.cache import InMemoryCache
import langchain

langchain.llm_cache = InMemoryCache()


def get_llm(model_name="text-ada-001", temperature=0.0):
    return OpenAI(
        model_name=model_name,
        temperature=temperature,
        model_kwargs=dict(logprobs=5, stop=["\n"]),
    )


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


def token_beams(response, prompt, strings=None, lprobs=None, cut_n=25, i=0):
    """Build full strings and probabilities from the token logprobs with beam search."""
    if strings is None:
        strings = [""]
    if lprobs is None:
        lprobs = [0]
    # If we're at the end of the string, return the strings and probs
    if i == len(response["logprobs"]["top_logprobs"]):
        # parse the strings into values
        if len(strings) == 0:
            return np.array([1.0]), np.array([0.0])
        values = dict()
        lmax = max(lprobs)
        for p, s in zip(lprobs, [truncate(s) for s in strings]):
            try:
                v = float(s)
                if v not in values:
                    values[v] = 0
                values[v] += np.exp(p - lmax)
            except ValueError:
                pass
        probs = np.array(list(values.values()))
        probs /= probs.sum()
        return probs, np.array(list(values.keys()))
    new_strings = []
    new_lprobs = []
    for value_str, logp_str in response["logprobs"]["top_logprobs"][i].items():
        try:
            lp = float(logp_str)
            if i == 0:
                # If we're at the beginning of the string, we need to remove the prompt
                value_str = remove_overlap(prompt, value_str)
            new_strings.extend([s + value_str for s in strings])
            new_lprobs.extend([p + lp for p in lprobs])
        except ValueError:
            print(f"ValueError: {value_str}, {logp_str}")
    # Filter out the low-probability strings
    cutoff = np.argsort(new_lprobs)[-cut_n:]
    new_strings = [new_strings[c] for c in cutoff]
    new_lprobs = [new_lprobs[c] for c in cutoff]
    return token_beams(
        response, prompt, strings=new_strings, lprobs=new_lprobs, cut_n=cut_n, i=i + 1
    )


def openai_predict(query_list, llm, *args, **kwargs):
    """Predict the output numbers for a given list of queries"""
    completion_response = llm.generate(query_list, *args, **kwargs)
    results = []
    for gen, q in zip(completion_response.generations, query_list):
        results.append(token_beams(gen[0].generation_info, q))
    return results
