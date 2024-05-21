import bolift
from bolift import (AskTellFewShotMulti,
                    AskTellFewShotTopk,
                    AskTellGPR,
                    AskTellNearestNeighbor,
                    AskTellRidgeKernelRegression,
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
        models = metafunc.cls.models_to_test()
        metafunc.parametrize("asktell_class", models)

class TestAskTell(ABC):
    __test__ = False

    def test_fewshot(self, asktell_class):
        asktell = asktell_class(x_formatter=lambda x: f"y = 2*{x}")
        asktell.tell(1, 2)
        asktell.tell(2, 4)
        asktell.tell(4, 8)
        asktell.tell(16, 32)

        dist = asktell.predict(2)
        m = dist.mode()
        assert m == pytest.approx(4, 0.001)
        
        dist = asktell.predict(3)
        m = dist.mode()
        assert m == pytest.approx(6, 0.001)

    def test_inv_fewshot(self, asktell_class):
        asktell = asktell_class(x_formatter=lambda x: f"2*{x}")
        asktell.tell(1, 2)
        asktell.tell(2, 4)
        asktell.tell(4, 8)
        asktell.tell(16, 32)

        inverse = asktell.inv_predict(8)
        assert "*" in inverse

class TestAskTellMulti(TestAskTell):
    __test__ = False

class TestAskTellTopK(TestAskTell):
    __test__ = True

    @pytest.fixture
    def asktell_class():
        return AskTellFewShotTopk
    
class TestAskTellKNN(TestAskTell):
    __test__ = False

class TestAskTellKRR(TestAskTell):
    __test__ = False

class TestAskTellGPR():
    __test__ = True

    @pytest.fixture
    def asktell_class():
        return AskTellGPR
    
    def test_gpr(asktell_class):
        asktell = asktell_class(
            x_formatter=lambda x: f"y = 2 + {x}",
            y_formatter=lambda y: str(int(y)),
        )
        for i in range(5):
            asktell.tell(i, 2 + i, train=False)
        asktell.tell(5, 7, train=True)
        asktell.predict(5000)
        assert asktell.ask([1, 5])[0][0] == 5

class TestAskTellFineTuning():
    __test__ = True

    @pytest.fixture
    def asktell_class():
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