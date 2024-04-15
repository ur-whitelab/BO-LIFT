import bolift
from bolift import llm_model
import numpy as np
from abc import ABC
import pytest

from langchain.schema import HumanMessage, SystemMessage

np.random.seed(0)

def pytest_generate_tests(metafunc):
    if 'model_name' in metafunc.fixturenames:
        models = metafunc.cls.models_to_test()
        metafunc.parametrize("model_name", models)

class TestLLM(ABC):
    __test__ = False

    @classmethod
    def models_to_test(cls):
        raise NotImplementedError("`models_to_test` must be implemented in subclasses.")

    def test_completion(self, model_name):
        llm = llm_model.get_llm(model_name=model_name, stop=["\n\n"])
        result, token = llm.predict("The value of 1 + 1 is")
        assert result[0].mean() == 2

class TestChatOpenAILLM(TestLLM):
    __test__ = True

    @classmethod
    def models_to_test(cls):
        return ["gpt-3.5-turbo"]
    
    def test_parse(self, model_name):
        llm = llm_model.get_llm(model_name=model_name, stop=["\n"])
        result, token = llm.predict("The value of 1 + 1 is")
        assert result[0].mean() == 2

class TestOpenAILLM(TestLLM):
    __test__ = True

    @classmethod
    def models_to_test(cls):
        return ["gpt-3.5-turbo-instruct"]

