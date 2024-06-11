import bolift
from bolift import (AskTellFewShotMulti,
                    AskTellFewShotTopk,
                    AskTellGPR,
                    AskTellNearestNeighbor,
                    # AskTellRidgeKernelRegression,
                    AskTellFinetuning)
import numpy as np
from abc import ABC
import pytest
import os
import openai
import time


np.random.seed(0)

def pytest_generate_tests(metafunc):
    if 'asktell_class' in metafunc.fixturenames:
        models = metafunc.cls.asktells_to_test()
        metafunc.parametrize("asktell_class", models)
    if 'model_name' in metafunc.fixturenames:
        models = metafunc.cls.models_to_test()
        metafunc.parametrize("model_name", models)


class TestAskTell(ABC):
    __test__ = False # Will only test in the children classes

    def test_tell(self, asktell_class):
        asktell = asktell_class(x_formatter=lambda x: f"y = {x}+2")
        for k in range(5):
            asktell.tell(k, k+2)
        assert asktell._example_count == 5

    def test_fewshot(self, asktell_class, model_name):
        asktell = asktell_class(model=model_name,
                                x_formatter=lambda x: f"y = 2*{x}"
                                )
        asktell.tell(1, 2)
        asktell.tell(2, 4)
        asktell.tell(4, 8)
        asktell.tell(16, 32)

        sys_msg = "You are a calculator. Given the prompt, provide the answer. Please complete the result only without any explanation."

        dist = asktell.predict(2, system_message=sys_msg)
        m = dist.mode()
        assert m == pytest.approx(4, 0.001)
        
        dist = asktell.predict(3, system_message=sys_msg)
        m = dist.mode()
        assert m == pytest.approx(6, 0.001)

    def test_inverse_fewshot(self, asktell_class, model_name):

        sys_msg = "You are a inverse calculator. Given the number in the prompt, your task is to provide a matemathical equation that generates that number. Please complete the mathematical equation only without any explanation."

        asktell = asktell_class(model=model_name,
                                x_formatter=lambda x: f"y = 2*{x}"
                                )
        asktell.tell(1, 2)
        asktell.tell(2, 4)
        asktell.tell(4, 8)
        asktell.tell(16, 32)

        inverse = asktell.inv_predict(8, system_message=sys_msg)
        assert "*" in inverse


class TestAskTellMulti(TestAskTell):
    __test__ = False


class TestAskTellTopK(TestAskTell):
    __test__ = True

    @classmethod
    def asktells_to_test(cls):
        return [AskTellFewShotTopk]
    
    @classmethod
    def models_to_test(cls):
        return ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo"]
    

class TestAskTellKNN(TestAskTell):
    __test__ = False


class TestAskTellKRR(TestAskTell):
    __test__ = False


class TestAskTellGPR():
    __test__ = True

    # @classmethod
    # def asktells_to_test(cls):
    #     return [AskTellGPR]
    
    # @pytest.mark.parametrize("asktell_class", [AskTellGPR])
    def test_gpr(self):
        asktell = AskTellGPR(
            x_formatter=lambda x: f"y = 2 + {x}",
            y_formatter=lambda y: str(int(y)),
        )
        for i in range(5):
            asktell.tell(i, 2 + i, train=False)
        asktell.tell(5, 7, train=True)
        asktell.predict(5000)
        assert asktell.ask([1, 5])[0][0] == 5


class TestAskTellFineTuning():
    __test__ = False

    @pytest.fixture
    def asktells_to_test():
        return AskTellFinetuning
    
    def test_prepare_data_finetuning(asktell_class):
        asktell = asktell_class(
            x_formatter=lambda x: f"y = 2 * {x}", model="text-ada-001", finetune=True
        )
        prompts = ["2", "4", "8"]
        completions = ["4", "8", "16"]
        asktell.prepare_data(prompts, completions, "./test.jsonl")
        assert os.path.exists("./test.jsonl")
        assert open("./test.jsonl").readlines() == [
            '{"prompt": "2", "completion": "4"}\n',
            '{"prompt": "4", "completion": "8"}\n',
            '{"prompt": "8", "completion": "16"}\n',
        ]
        os.remove("./test.jsonl")


    def test_upload_data_finetuning(asktell_class):
        asktell = asktell_class(
            x_formatter=lambda x: f"y = 2 * {x}", model="text-ada-001", finetune=True
        )
        prompts = ["2", "4", "8"]
        completions = ["4", "8", "16"]
        asktell.prepare_data(prompts, completions, "./test.jsonl")
        file_id = asktell.upload_data("./test.jsonl")
        time.sleep(30)  # Sometimes it take a few seconds for the file to be uploaded
        assert file_id is not None
        assert (
            openai.File.retrieve(file_id).status == "uploaded"
            or openai.File.retrieve(file_id).status == "processed"
        )
        os.remove("./test.jsonl")
        openai.File.delete(file_id)