from bolift import llm_model
import bolift
import numpy as np

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
    assert abs(result.mode().astype(int) - 4) <= 1


def test_tell_fewshot():
    asktell = bolift.AskTellFewShotMulti(
        x_formatter=lambda x: f"y = 2 * {x}",
        y_formatter=lambda y: str(int(y)),
        verbose=True,
    )
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    asktell.tell(3, 6)
    dist = asktell.predict(3)
    dist.mode()
    # assert m == 6


def test_tell_inv_fewshot():
    asktell = bolift.AskTellFewShotMulti(
        x_formatter=lambda x: f"2 * {x}",
        y_formatter=lambda y: str(int(y)),
        verbose=True,
    )
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    asktell.tell(3, 6)
    inverse = asktell.inv_predict(6)
    assert "*" in inverse


def test_tell_inv_fewshot_topk():
    asktell = bolift.AskTellFewShotTopk(
        x_formatter=lambda x: f"2 * {x}",
        y_formatter=lambda y: str(int(y)),
        verbose=True,
    )
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    asktell.tell(3, 6)
    inverse = asktell.inv_predict(6)
    assert "*" in inverse


def test_tell_fewshot_topk():
    asktell = bolift.AskTellFewShotTopk(x_formatter=lambda x: f"y = 2*{x}")
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    asktell.tell(16, 32)
    dist = asktell.predict(2)
    m = dist.mode().astype(int)
    assert m == 4


def test_tell_fewshot_vark_topk():
    asktell = bolift.AskTellFewShotTopk(
        x_formatter=lambda x: f"y = 2 * {x}", k=1, y_formatter=lambda y: str(int(y))
    )
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    dist = asktell.predict(2)
    m = dist.mode()
    assert m == 4


def test_tell_fewshot_vark():
    asktell = bolift.AskTellFewShotMulti(
        x_formatter=lambda x: f"y = 2*{x}",
        k=3,
        verbose=True,
        y_formatter=lambda y: str(int(y)),
    )
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    asktell.tell(3, 6)
    dist = asktell.predict(2)
    dist.mode()
    # assert m == 4


def test_tell_fewshot_selector():
    asktell = bolift.AskTellFewShotMulti(
        x_formatter=lambda x: f"y = 2 + {x}",
        selector_k=3,
        y_formatter=lambda y: str(int(y)),
        verbose=True,
    )
    for i in range(5):
        asktell.tell(i, 2 + i)
    dist = asktell.predict(3)
    assert abs(3 + 2 - dist.mode().astype(int)) < 10


def test_tell_fewshot_selector_less():
    asktell = bolift.AskTellFewShotMulti(
        x_formatter=lambda x: f"y = 2 + {x}",
        selector_k=10,
        y_formatter=lambda y: str(int(y)),
        verbose=True,
    )
    for i in range(5):
        asktell.tell(i, 2 + i)
    dist = asktell.predict(3)
    assert abs(3 + 2 - dist.mode().astype(int)) < 10


def test_tell_fewshot_topk_selector():
    asktell = bolift.AskTellFewShotTopk(
        x_formatter=lambda x: f"y = 2 + {x}",
        selector_k=3,
        y_formatter=lambda y: str(int(y)),
        verbose=True,
    )
    for i in range(5):
        asktell.tell(i, 2 + i)
    dist = asktell.predict(3)
    assert abs(3 + 2 - dist.mode().astype(int)) < 10


def test_ask_ei_fewshot():
    asktell = bolift.AskTellFewShotMulti(x_formatter=lambda x: f"y = 2 * {x}")
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    asktell.tell(16, 32)
    best, _, _ = asktell.ask([2, 8])
    assert best[0] == 8


def test_ask_ei_fewshot_pool():
    asktell = bolift.AskTellFewShotMulti(x_formatter=lambda x: f"y = 2 * {x}")
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    asktell.tell(16, 32)
    pool = list(range(30))
    best, _, _ = asktell.ask(pool, inv_filter=3)
    assert best[0] > 10


def test_ask_ei_fewshot_pool_topk():
    asktell = bolift.AskTellFewShotTopk(x_formatter=lambda x: f"y = 2 * {x}")
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    asktell.tell(16, 32)
    pool = list(range(30))
    best, _, _ = asktell.ask(pool, inv_filter=3)
    assert best[0] > 10


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
    assert scores[0] >= scores[1]
    asktell.ask([2, 8], k=2, aq_fxn="greedy")


def test_ask_ei_fewshot_finetune():
    asktell = bolift.AskTellFinetuning(
        x_formatter=lambda x: f"y = 2 * {x}",
        finetune=False,
    )
    asktell.tell(2, 4)
    asktell.tell(1, 2)
    asktell.tell(16, 32)
    best, _, _ = asktell.ask([2, 8])
    assert best[0] == 8


def test_prepare_data_finetuning():
    import os

    asktell = bolift.AskTellFinetuning(
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


def test_upload_data_finetuning():
    import os
    import openai
    import time

    asktell = bolift.AskTellFinetuning(
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


def test_llm_usage():
    asktell = bolift.AskTellFewShotMulti(
        x_formatter=lambda x: f"y = 2 + {x}",
        selector_k=10,
        y_formatter=lambda y: str(int(y)),
    )
    for i in range(5):
        asktell.tell(i, 2 + i)
    asktell.predict(3)
    print(asktell.tokens_used)
    assert asktell.tokens_used > 0


def test_gpr():
    asktell = bolift.AskTellGPR(
        x_formatter=lambda x: f"y = 2 + {x}",
        y_formatter=lambda y: str(int(y)),
    )
    for i in range(5):
        asktell.tell(i, 2 + i, train=False)
    asktell.tell(5, 7, train=True)
    asktell.predict(5000)
    assert asktell.ask([1, 5])[0][0] == 5
