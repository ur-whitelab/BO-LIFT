import os
from typing import Dict, List, Tuple
import openai
import pandas as pd
import time
import json

from typing import *
from langchain.prompts.prompt import PromptTemplate
from .llm_model import get_llm
import warnings


class AskTellFinetuning:
    def __init__(
        self,
        model: str = "gpt-4o-mini-2024-07-18",
        temperature: Optional[float] = 0.7,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        prompt_template: PromptTemplate = None,
        x_formatter: Callable[[str], str] = lambda x: x,
        y_formatter: Callable[[float], str] = lambda y: f"{y:0.2f}",
        y_name: str = "y",
        verbose: bool = False,
        id: str = None,
        n_epochs: int = 8,
        learning_rate: int = 0.02,
        examples: List[Tuple[str, float]] = [],
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.prefix = prefix
        self.suffix = suffix
        self.prompt_template = prompt_template
        self.y_name = y_name
        self.x_formatter = x_formatter
        self.y_formatter = y_formatter
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        # self.model = model.split("-")[1]
        self.examples = examples
        self._example_count = 0
        self.tokens_used = 0
        self.response = None
        self.llm = self._setup_llm(model=self.model)
        self.client = openai.OpenAI()
        if id:
            self.response = self.client.fine_tuning.jobs.retrieve(id)
            self.model = self.response.fine_tuned_model
            self.llm = self._setup_llm(model=self.model)

    def prepare_data_from_file(self, inFile, outFile):
        reg_df = pd.read_csv(inFile)
        with open(outFile, "w") as f:
            for e in range(len(reg_df)):
                f.write(
                    f'{{"prompt":"{reg_df.iloc[e]["prompt"]}", "completion":"{reg_df.iloc[e]["completion"]}"}}\n'
                )

    def prepare_data(self, prompts, completions, outFile):
        with open(outFile, "w") as f:
            for p, c in zip(prompts, completions):
                messages = {
                    "messages": [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": self.x_formatter(p)},
                        {"role": "assistant", "content": self.y_formatter(c)},
                    ]
                }
                json_string = json.dumps(messages)
                f.write(f"{json_string}\n")
                # Deprecated chat model. Saving for reference
                # f.write(f'{{"prompt": "{p}", "completion": "{c}"}}\n')

    def upload_data(self, data):
        f = self.client.files.create(
            file=open(data, "rb"),
            purpose="fine-tune",
        )
        file_id = f.id
        return file_id

    def create_fine_tune(self, file_id, model, n_epochs=8, learning_rate=0.02):
        response = self.client.fine_tuning.jobs.create(
            training_file=file_id,
            model=model,
            # hyperparameters={
            #     "n_epochs": n_epochs,
            #     "learning_rate": learning_rate,
            # }
        )
        return response

    def fine_tune(self, prompts, completions, out_path="./out", out_file=None) -> None:
        if not os.path.exists(f"{out_path}"):
            os.makedirs(f"{out_path}")
        self.prepare_data(
            prompts,
            completions,
            f"{out_path}/train_data_{len(prompts)}.jsonl",
        )
        file_id = self.upload_data(f"{out_path}/train_data_{len(prompts)}.jsonl")
        response = self.create_fine_tune(
            file_id, self.model, self.n_epochs, self.learning_rate
        )

        s = self.client.fine_tuning.jobs.retrieve(response.id).status
        t = 0
        print("\n")
        while s != "succeeded":
            if t % 3 == 0:
                s += ".   "
            elif t % 3 == 1:
                s += "..  "
            elif t % 3 == 2:
                s += "... "
            event_message = (
                self.client.fine_tuning.jobs.list_events(fine_tuning_job_id=response.id)
                .data[-1]
                .message
            )
            s += f"{event_message}                                     "
            print(s, end="\r")

            s = self.client.fine_tuning.jobs.retrieve(response.id).status
            t += 1
            time.sleep(2)
        print("\n")
        event_message = (
            self.client.fine_tuning.jobs.list_events(fine_tuning_job_id=response.id)
            .data[-1]
            .message
        )
        print(f"{s}... {event_message}")

        self.response = self.client.fine_tuning.jobs.retrieve(response.id)
        self.model = self.response.fine_tuned_model
        self.llm = self._setup_llm(model=self.response.fine_tuned_model)
        if out_file:
            with open(f"{out_path}/{out_file}.dat", "w") as out:
                out.write(json.dumps(self.response, indent=4))
        else:
            with open(f"{out_path}/FT_model_{len(self.examples)}.dat", "w") as out:
                out.write(json.dumps(dict(self.response).__str__(), indent=4))

    def get_model_name(self):
        return self.model
        # return self.response["fine_tuned_model"] if self.response else self.model

    def _setup_llm(self, model: str):
        return get_llm(model_name=model, temperature=self.temperature, n=1)

    def ask(*args, **kwargs):
        raise NotImplementedError("Finetuned models does not support ask method")

    def tell(
        self, x: str, y: float, alt_ys: Optional[List[float]] = None, train=False
    ) -> None:
        self.examples.append((x, y))
        self._example_count += 1

        if train:
            self.fine_tune(
                [x[0] for x in self.examples],
                [x[1] for x in self.examples],
            )

    def predict(
        self, x: str, system_message: Optional[str] = ""
    ) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        if not isinstance(x, list):
            x = [x]

        queries = [x_i for x_i in x]

        if not system_message:
            warnings.warn(
                "No system message provided for prediction. Using default. \nNot clearly specifying the task for the LLM usually decreases its performance considerably."
            )

        results, tokens = self.llm.predict(queries, system_message=system_message)
        self.tokens_used += tokens
        if len(x) == 1:
            return results[0]
        return results

    def _setup_prompt(
        self,
        example: Dict,
        prompt_template: Optional[PromptTemplate] = None,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> PromptTemplate:
        self.prompt = PromptTemplate(
            input_variables=["x", "y", "y_name"],
            template="Q: Given {x}, what is {y_name}?\nA: {y}###\n\n",
        )
