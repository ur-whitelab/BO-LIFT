import os
import openai
import numpy as np
from functools import partial
import pandas as pd
from pyrate_limiter import Duration, Limiter, RequestRate
import time
import json

from typing import *
from .asktell import AskTellFewShot
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

from .llm_model import (
    get_llm, 
    openai_choice_predict, 
    openai_topk_predict, 
    DiscreteDist,
)
from .aqfxns import (
    probability_of_improvement,
    expected_improvement,
    upper_confidence_bound,
    greedy,
)

class AskTellFinetuning(AskTellFewShot):
    def __init__(
        self,
        prompt_template: PromptTemplate = None,
        suffix: Optional[str] = None,
        model: str = "text-curie-001",
        prefix: Optional[str] = None,
        x_formatter: Callable[[str], str] = lambda x: x,
        y_formatter: Callable[[float], str] = lambda y: f"{y: 0.2f}",
        y_name: str = "y",
        id: str = None,
        n_epochs: int = 2,
        batch_size: int = 1,
        finetune: bool = False
    ) -> None:
        super().__init__(prompt_template, suffix, model, prefix, x_formatter, y_formatter, y_name)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.response = None
        self.finetune = finetune
        self.base_model = model.split("-")[1]
        if(id):
            self.response = openai.FineTune.retrieve(id=id)
            self.base_model = self.response["fine_tuned_model"]
        self.llm = self._setup_llm(model=self.base_model)


    def prepare_data_from_file(self, inFile, outFile):
        reg_df = pd.read_csv(inFile)
        # reg_df.to_json("regression_dataset_sial_ratio.jsonl", orient='records', lines=True)
        with open(outFile, "w") as f:
            for e in range(len(reg_df)):
                f.write(f'{{"prompt": "{reg_df.iloc[e]["prompt"]}", "completion": "{reg_df.iloc[e]["completion"]}"}}\n')

    def prepare_data(self, prompts, completions, outFile):
        with open(outFile, "w") as f:
            for p,c in zip(prompts, completions):
                f.write(f'{{"prompt": "{p}", "completion": "{c}"}}\n')

            
    def upload_data(self, data):
        f = openai.File.create(
          file=open(data, "rb"),
          purpose='fine-tune',
          user_provided_filename="FTimplementation.jsonl"
        )
        file_id = f["id"]
        return file_id


    def create_fineTune(self, file_id, base_model):
        response = openai.FineTune.create(
                              training_file=file_id, 
                              model=base_model,
                              n_epochs=self.n_epochs,
                              batch_size=self.batch_size, 
                            )
        return response


    def fineTune(self, prompts, completions) -> None:
        out_path = "out"
        if not os.path.exists(f"{out_path}"):
            os.makedirs(f"{out_path}")
        self.prepare_data(prompts,
                          completions,
                          f"{out_path}/train_data_{len(self.prompt.examples)}.jsonl"
                        )
        file_id = self.upload_data(f"{out_path}/train_data_{len(self.prompt.examples)}.jsonl")

        print(self.base_model)
        response = self.create_fineTune(file_id, self.base_model)

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
            out.write(json.dumps(response, indent=4))


    def get_model_id(self):
        return self.response["fine_tuned_model"]


    # def _setup_prompt(
    #     self,
    #     prompt_template: Optional[PromptTemplate] = None,
    #     suffix: Optional[str] = None,
    #     prefix: Optional[str] = None,
    # ) -> FewShotPromptTemplate:
    #     if prefix is None:
    #         prefix = (
    #             "Answer correctly the following questions.\n"
    #             "Your answers must be numerical and "
    #             "you should end your answer with ###\n\n"
    #         )
    #     if prompt_template is None:
    #         prompt_template = PromptTemplate(
    #             input_variables=["x", "y", "i", "y_name"],
    #             template="Q{i}: Given {x}, what is {y_name}?\nA{i}: {y}###\n\n",
    #         )
    #         if suffix is not None:
    #             raise ValueError(
    #                 "Cannot provide suffix if using default prompt template."
    #             )
    #         suffix = "Q{i}: Given {x}, what is {y_name}?\nA{i}: "
    #     elif suffix is None:
    #         raise ValueError("Must provide suffix if using custom prompt template.")
    #     # test out prompt
    #     prompt_template.format(x="test", y="2.3", i=1, y_name="tee")
    #     return FewShotPromptTemplate(
    #         examples=[],
    #         example_prompt=prompt_template,
    #         suffix=suffix,
    #         prefix=prefix,
    #         input_variables=["x", "i", "y_name"],
    #     )


    # def _predict(self, queries: List[str]) -> List[DiscreteDist]:
    #     return openai_topk_predict(queries, self.llm)


    def predict(self, xs):
        xs = xs if isinstance(xs, list) else [xs]
        queries = [
            self.prompt.format(
                x=self._x_formatter(x_i),
                i=len(self.prompt.examples) + 1,
                y_name=self._y_name,
            )
            for x_i in xs
        ]
        print(self.base_model)
        if(self.finetune):
            # Maybe run this only when we have a few new prompts? Each 5 new points??
            if len(self.prompt.examples)%2 == 0 and len(self.prompt.examples)>0:
                prompts = [p['x'] for p in self.prompt.examples]
                completions = [p[p['Answer']] for p in self.prompt.examples]
                self.fineTune(prompts, completions)

            # 2 points -> ft-H3Bx697RyEZYZ7OAhtcPUnEI
            # 4 points -> ft-h3p6ILhhtNCi1QFDsGB89amC
            # self.response = openai.FineTune.retrieve(id="ft-H3Bx697RyEZYZ7OAhtcPUnEI")
            # self.llm = self._setup_llm(model=self.response["fine_tuned_model"])
        results = self._predict(queries)
        return results