from bolift import llm_model
import bolift
import dataclasses
import numpy as np
from langchain.prompts.prompt import PromptTemplate


def test_remove_overlap():
    s1 = "This is a test"
    s2 = "a test of the emergency broadcast system"
    assert llm_model.remove_overlap(s1, s2) == " of the emergency broadcast system"
    s1 = "This is a test"
    s2 = " to make sure there is none"
    assert llm_model.remove_overlap(s1, s2) == s2


def test_truncate():
    assert llm_model.truncate("1.0,") == "1.0"
    assert llm_model.truncate("1.0") == "1.0"
    assert llm_model.truncate("1.0\n") == "1.0"
    assert llm_model.truncate("1.0\n\n") == "1.0"
    assert llm_model.truncate("-1243.433+3433") == "-1243.433"


def test_beam_search():
    # response.choices[0]["logprobs"]["top_logprobs"]
    tokens = [{"=1": -1.0}, {"\n": -2.0, "2\n": -1.0}]

    mock_response = dict(logprobs=dict(top_logprobs=tokens))
    lprobs_expected = np.array([-3, -2])
    values_expected = np.array([1, 12])
    max = lprobs_expected.max()
    probs_expected = np.exp(lprobs_expected - max)
    probs_expected /= probs_expected.sum()
    probs, values = llm_model.token_beams(mock_response, "1+1=")
    assert np.allclose(probs, probs_expected)
    assert np.allclose(values, values_expected)


def test_completion():
    llm = llm_model.get_llm()
    assert llm("The value of 1 + 1 is") == " 2"


def test_tell():
    template = PromptTemplate(
        input_variables=["x", "y"],
        template="Q: What is the value of 2 * {x}?\n" "A: {y}\n",
    )
    asktell = bolift.AskTellFewShot(
        template, suffix="Q: What is the value of 2 * {x}?\n A:"
    )
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    asktell.tell(16, 32)
    probs, values = asktell.predict(2)
    m = values[np.argmax(probs)]
    assert m == 4


def test_ask_ei():
    template = PromptTemplate(
        input_variables=["x", "y"],
        template="Q: What is the value of 2 * {x}?\n" "A: {y}\n",
    )
    asktell = bolift.AskTellFewShot(
        template, suffix="Q: What is the value of 2 * {x}?\n A:"
    )
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    asktell.tell(16, 32)
    best, _, _ = asktell.ask(["2", "8"])
    assert best[0] == "8"


def test_ask_zero():
    template = PromptTemplate(
        input_variables=["x", "y"],
        template="Q: What is the value of 2 * {x}?\n" "A: {y}\n",
    )
    asktell = bolift.AskTellFewShot(
        template, suffix="Q: What is the value of 2 * {x}?\n A:"
    )
    _ = asktell.ask(
        ["2", "8"],
    )

    _, scores, _ = asktell.ask(["2", "8"], k=2)
    assert scores[0] > scores[1]

    asktell.ask(["2", "8"], k=2, aq_fxn="greedy")
