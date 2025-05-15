import boicl
from boicl import (
    AskTellFewShotTopk,
    AskTellGPR,
    AskTellNearestNeighbor,
    AskTellRidgeKernelRegression,
    AskTellFinetuning,
    Pool,
)
import numpy as np
from abc import ABC
import pytest
import os
import openai
import time


np.random.seed(0)


def pytest_generate_tests(metafunc):
    if "asktell_class" in metafunc.fixturenames:
        models = metafunc.cls.asktells_to_test()
        metafunc.parametrize("asktell_class", models)
    if "model_name" in metafunc.fixturenames:
        models = metafunc.cls.models_to_test()
        metafunc.parametrize("model_name", models)


class TestAskTell(ABC):
    __test__ = False  # Will only test in the children classes

    def test_tell(self, asktell_class):
        asktell = asktell_class(x_formatter=lambda x: f"y = {x}+2")
        for k in range(5):
            asktell.tell(k, k + 2)
        assert asktell._example_count == 5

    def test_fewshot(self, asktell_class, model_name):
        asktell = asktell_class(model=model_name, x_formatter=lambda x: f"y = 2*{x}")
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

        sys_msg = "You are a inverse calculator. Given the number in the prompt, your task is to provide a mathematical equation that generates that number. Please complete the mathematical equation only without any explanation."

        asktell = asktell_class(model=model_name, x_formatter=lambda x: f"y = 2*{x}")
        asktell.tell(1, 2)
        asktell.tell(2, 4)
        asktell.tell(4, 8)
        asktell.tell(16, 32)

        inverse = asktell.inv_predict(8, system_message=sys_msg)
        assert "*" in inverse

    def test_example_counter(self, asktell_class, model_name):
        asktell = asktell_class(
            model=model_name,
            x_formatter=lambda x: f"y = 2+{x}",
            y_formatter=lambda y: str(int(y)),
        )
        assert asktell._example_count == 0
        assert asktell._ready == False

        asktell.tell(0, 2)
        asktell.tell(1, 3)
        asktell.tell(2, 4)

        assert asktell._example_count == 3
        assert asktell._ready == True

    def test_llm_usage(self, asktell_class):
        asktell = asktell_class(
            x_formatter=lambda x: f"y = 2 + {x}",
            selector_k=10,
            y_formatter=lambda y: str(int(y)),
        )
        for i in range(5):
            asktell.tell(i, 2 + i)
        asktell.predict(3)
        assert asktell.tokens_used > 0


class TestAskTellTopK(TestAskTell):
    __test__ = True

    @classmethod
    def asktells_to_test(cls):
        return [AskTellFewShotTopk]

    @classmethod
    def models_to_test(cls):
        return ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo"]


class TestAskTellKNN:
    __test__ = True

    # @classmethod
    # def asktells_to_test(cls):
    #     return [AskTellNearestNeighbor]
    # @classmethod
    # def models_to_test(cls):
    #     return ["gpt-3.5-turbo"]

    def test_knn(self):
        asktell = AskTellNearestNeighbor(
            x_formatter=lambda x: f"y = 2+{x}",
            y_formatter=lambda y: str(int(y)),
        )

        asktell.tell(0, 2)
        asktell.tell(1, 3)
        asktell.tell(3, 5)

        dist = asktell.predict(2)
        m = dist.mode()
        assert m == pytest.approx(3.333, 0.01)
        asktell.tell(5, 7)
        asktell.tell(6, 8)
        dist = asktell.predict(4)
        m = dist.mode()
        assert abs(m - 6) <= 1.0


class TestAskTellKRR:
    __test__ = True

    def test_krr(self):
        asktell = AskTellRidgeKernelRegression(
            x_formatter=lambda x: f"y = 2+{x}",
            y_formatter=lambda y: str(int(y)),
        )
        for i in range(5):
            asktell.tell(i, 2 + i)
        asktell.tell(5, 7)
        assert len(asktell.examples) == 6
        assert asktell._example_count == 6

        m = asktell.predict(5)
        assert m.mean() == pytest.approx(7, 0.1)
        assert asktell.ask([1, 5, 8], k=1)[0][0] in [1, 5, 8]


class TestAskTellGPR:
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
        asktell.tell(6, 8, train=True)
        assert len(asktell.examples) == 6
        assert asktell._example_count == 6

        m = asktell.predict(5)
        assert m.mean() == pytest.approx(7, 0.1)
        assert asktell.ask([1, 5, 8], k=1)[0][0] in [1, 5, 8]

    def test_gpr_fail_train(self):
        asktell = AskTellGPR(
            x_formatter=lambda x: f"y = 2 + {x}",
            y_formatter=lambda y: str(int(y)),
        )
        with pytest.raises(ValueError):
            asktell.tell(2, 2 + 2)

    def test_gpr_train_w_pool(self):
        pool = Pool(["1", "2", "3", "4", "5", "6"])
        asktell = AskTellGPR(
            x_formatter=lambda x: f"y = 2 + {x}",
            y_formatter=lambda y: str(int(y)),
            pool=pool,
        )
        asktell.tell(2, 2 + 2)
        assert asktell.predict(2).mean() == pytest.approx(4, 0.05)
        assert asktell.ask([1, 2], k=1)[0][0] in [1, 2]


class TestAskTellFineTuning:
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
