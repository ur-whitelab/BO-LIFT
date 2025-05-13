import boicl
from boicl import llm_model
import numpy as np
from abc import ABC
import pytest

from langchain.schema import HumanMessage, SystemMessage

np.random.seed(0)


def pytest_generate_tests(metafunc):
    if "model_name" in metafunc.fixturenames:
        models = metafunc.cls.models_to_test()
        metafunc.parametrize("model_name", models)


class TestLLM(ABC):
    __test__ = False

    @classmethod
    def models_to_test(cls):
        raise NotImplementedError("`models_to_test` must be implemented in subclasses.")

    def test_completion(self, model_name):
        llm = llm_model.get_llm(model_name=model_name, stop=["\n\n"])
        result, token = llm.predict("How much is 1 + 1? Answer the only the number")
        assert result[0].mean() == pytest.approx(2, 0.1)


class TestOpenAILLM(TestLLM):
    __test__ = True

    @classmethod
    def models_to_test(cls):
        return ["gpt-3.5-turbo-instruct"]


class TestChatOpenAILLM(TestLLM):
    __test__ = True

    @classmethod
    def models_to_test(cls):
        return ["gpt-3.5-turbo-0125"]

    def test_parse_response(self, model_name):
        print(model_name)
        prompt = "2 + 2 is"
        answer = 4
        llm = llm_model.get_llm(
            n=3,
            best_of=3,
            temperature=1,
            model_name=model_name,
            top_p=0.99,
            stop=["\n"],
        )

        query = [SystemMessage(content=""), HumanMessage(content=prompt)]

        g = llm.llm.generate([query]).generations
        result = llm.parse_response(g[0])

        assert abs(result.mode().astype(int) - answer) <= 1
