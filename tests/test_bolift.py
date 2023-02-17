from bolift import llm_model
import bolift
import dataclasses
import numpy as np
from langchain.prompts.prompt import PromptTemplate

np.random.seed(0)


def test_completion():
    llm = llm_model.get_llm(stop=["\n\n"])
    assert llm("The value of 1 + 1 is") == " 2"


def test_parse_response():
    prompt = """
Problem 1: What is 4 x 5?
A. 12
B. 32
C. 20
D. 16
E. 24

Answer: C

Problem 2: What is 4 x 4?
"""
    llm = llm_model.get_llm(logprobs=5)
    generation = llm.generate([prompt]).generations[0][0]
    result = llm_model.parse_response(generation, prompt, llm)
    # make sure answer is max
    assert 16 in result.values.astype(int)


def test_parse_response_topk():
    prompt = "2 + 2 is"
    llm = llm_model.get_llm(
        n=3,
        best_of=3,
        temperature=1,
        model_name="text-babbage-001",
        top_p=0.99,
        stop=["\n"],
        logprobs=1,
    )
    g = llm.generate([prompt]).generations
    result = llm_model.parse_response_topk(g[0])
    # make sure answer is max
    assert 4 in result.values.astype(int)


def test_tell_fewshot():
    asktell = bolift.AskTellFewShotMulti(
        x_formatter=lambda x: f"y = 2 * {x}", y_formatter=lambda y: str(int(y))
    )
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    asktell.tell(16, 32)
    dist = asktell.predict(3)
    m = dist.mode()
    assert m == 9


def test_tell_fewshot_topk():
    asktell = bolift.AskTellFewShotTopk(x_formatter=lambda x: f"y = 2 * {x}")
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    asktell.tell(16, 32)
    dist = asktell.predict(2)
    m = dist.mode()
    assert m == 4


def test_tell_fewshot_selector():
    asktell = bolift.AskTellFewShotMulti(
        x_formatter=lambda x: f"y = 2 * {x}",
        selector_k=3,
        y_formatter=lambda y: str(int(y)),
    )
    for i in range(5):
        asktell.tell(i, 2 * i)
    dist = asktell.predict(3)
    assert 3 * 2 in dist.values


def test_ask_ei_fewshot():
    asktell = bolift.AskTellFewShotMulti(x_formatter=lambda x: f"y = 2 * {x}")
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    asktell.tell(16, 32)
    best, _, _ = asktell.ask([2, 8])
    assert best[0] == 8


def test_ask_ei_fewshot_topk():
    asktell = bolift.AskTellFewShotTopk(x_formatter=lambda x: f"y = 2 * {x}")
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    asktell.tell(16, 32)
    best, _, _ = asktell.ask([2, 8])
    assert best[0] == 8


def test_ask_zero_fewshot():
    asktell = bolift.AskTellFewShotMulti(x_formatter=lambda x: f"y = 2 * {x}")
    _, scores, _ = asktell.ask([2, 8], k=2)
    assert scores[0] > scores[1]
    asktell.ask([2, 8], k=2, aq_fxn="greedy")


def test_ask_zero_fewshot_topk():
    asktell = bolift.AskTellFewShotTopk(
        x_formatter=lambda x: f"y = f({x}) for the shifted Ackley function",
        model="text-babbage-001",
    )
    _, scores, means = asktell.ask([2, 8], k=2)
    assert scores[0] > scores[1]
    asktell.ask([2, 8], k=2, aq_fxn="greedy")
