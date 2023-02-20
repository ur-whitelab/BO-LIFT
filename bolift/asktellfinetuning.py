import os
import openai
import numpy as np
import pandas as pd
import time
import json

from typing import *
from .asktell import AskTellFewShotTopk
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate


class AskTellFinetuning(AskTellFewShotTopk):
    def __init__(
        self,
        prompt_template: PromptTemplate = None,
        suffix: Optional[str] = None,
        model: str = "text-curie-001",
        temperature: Optional[float] = None,
        prefix: Optional[str] = None,
        x_formatter: Callable[[str], str] = lambda x: x,
        y_formatter: Callable[[float], str] = lambda y: f"{y:0.2f}",
        y_name: str = "y",
        selector_k: Optional[int] = None,
        k: int = 5,
        verbose: bool = False,
        id: str = None,
        n_epochs: int = 4,
        batch_size: int = 1,
        finetune: bool = False,
        examples: List[Tuple[str, float]] = [],
    ) -> None:
        super().__init__(prompt_template, suffix, model, temperature, prefix, x_formatter, y_formatter, y_name, selector_k, k, verbose)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.response = None
        self.finetune = finetune
        self.base_model = model.split("-")[1]
        self.examples = examples
        if(id):
            self.response = openai.FineTune.retrieve(id=id)
            self.base_model = self.response["fine_tuned_model"]
        # self.llm = self._setup_llm(model=self.base_model)


    def prepare_data_from_file(self, inFile, outFile):
        reg_df = pd.read_csv(inFile)
        with open(outFile, "w") as f:
            for e in range(len(reg_df)):
                f.write(f'{{"prompt":"{reg_df.iloc[e]["prompt"]}", "completion":"{reg_df.iloc[e]["completion"]}"}}\n')

    def prepare_data(self, prompts, completions, outFile):
        with open(outFile, "w") as f:
            for p,c in zip(prompts, completions):
                p = p.replace('°', '')
                f.write(f'{{"prompt": "{p}", "completion": "{c}"}}\n')

            
    def upload_data(self, data):
        f = openai.File.create(
          file=open(data, "rb"),
          purpose='fine-tune',
          user_provided_filename="FTimplementation.jsonl"
        )
        file_id = f["id"]
        return file_id


    def create_fine_tune(self, file_id, base_model):
        response = openai.FineTune.create(
                              training_file=file_id, 
                              model=base_model,
                              n_epochs=self.n_epochs,
                              batch_size=self.batch_size, 
                            )
        return response


    def fine_tune(self, prompts, completions, out_path="./out") -> None:
        if not os.path.exists(f"{out_path}"):
            os.makedirs(f"{out_path}")
        self.prepare_data(prompts,
                          completions,
                          f"{out_path}/train_data_{len(self.prompt.examples)}.jsonl"
                        )
        file_id = self.upload_data(f"{out_path}/train_data_{len(self.prompt.examples)}.jsonl")

        response = self.create_fine_tune(file_id, self.base_model)

        s = openai.FineTune.retrieve(id=response["id"]).status
        t=0
        while (s != "succeeded"):
            if t%3 == 0:
                s+=".   "
            elif t%3 == 1:
                s+="..  "
            elif t%3 == 2:
                s+="... "
            event_message = openai.FineTune.retrieve(id=response["id"]).events[-1]["message"]
            s += f"{event_message}                                     "
            print(s, end='\r')
            
            s = openai.FineTune.retrieve(id=response["id"]).status
            t+=1
            time.sleep(2)
        print("\n")

        self.response = openai.FineTune.retrieve(id=response["id"])
        self.base_model = self.response["fine_tuned_model"]
        self.llm = self._setup_llm(model=self.response["fine_tuned_model"])

        with open(f"{out_path}/FT_model_{len(self.prompt.examples)}.dat", "w") as out:
            out.write(json.dumps(self.response, indent=4))


    def get_model_name(self):
        return self.response["fine_tuned_model"] if self.response else self.base_model

    def _tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> Dict:
        """Tell the optimizer about a new example."""
        if alt_ys is not None:
            raise ValueError("Alt ys not supported for topk.")
        
        if(self.finetune):
            if len(self.prompt.examples)%5 == 0 and len(self.prompt.examples)>0:
                self.examples.extend(self.prompt.examples)
                prompts = [f"Q: Given {p['x']}. What is {self._y_name}?\\nA: "
                            for p in self.examples]
                completions = [p['y'] for p in self.examples]
                self.fine_tune(prompts, completions)
                self.prompt.examples = None

        example_dict = dict(
            x=self._x_formatter(x),
            y=self._y_formatter(y),
            y_name=self._y_name,
        )
        self._ys.append(y)
        return example_dict
    
