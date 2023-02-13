import os
import openai
import numpy as np
from functools import partial
import pandas as pd
from pyrate_limiter import Duration, Limiter, RequestRate
import time
import json

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

class AskTellFinetuning:
    def __init__(
        self,
        data: str,
        base_model: str,
        n_epochs: int = 2, 
        batch_size: int = 1
    ):
        self.data = data
        self.llm = get_llm(model=base_model, logprobs=5)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.response = None


    def prepare_data(self, inFile, outFile):
        reg_df = pd.read_csv(inFile)
        # reg_df.to_json("regression_dataset_sial_ratio.jsonl", orient='records', lines=True)
        with open(outFile, "w") as f:
            for e in range(len(reg_df)):
                f.write(f'{{"prompt": "{reg_df.iloc[e]["prompt"]}", "completion": "{reg_df.iloc[e]["completion"]}"}}\n')

            
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


    def fineTune(self):
        file_id = self.upload_data(self.data)
        response = self.create_fineTune(file_id, self.base_model)
        print(response["id"])
        s = openai.FineTune.retrieve(id=response["id"]).status
        t=0
        while (s != "succeeded"):
            time.sleep(5)
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
        print("\n")
        self.response = response
        return response
    

    def get_model_id(self):
        return self.response["fine_tuned_model"]


    def _predict(self, prompt, max_tokens=8, temperature=0):
        # completion_response = openai.Completion.create(
        #                         model=self.response["fine_tuned_model"] if self.response else "davinci",
        #                         prompt=prompt,
        #                         max_tokens=max_tokens,
        #                         temperature=temperature,
        #                         logprobs=5,
        #                         # Should I add stop and logit_bias?
        #                     )
        return openai_choice_predict(prompt, self.llm)


    def predict(self, xs):
        results = []
        xs = xs if isinstance(xs, list) else [xs]
        for _, x in enumerate(xs):
            results.append(self._predict(x))
        return results


    def tell():
        pass


    def ask(self, possible_x, aq_fxn = "upper_confidence_bound", _lambda=0.5):
        if aq_fxn == "probability_of_improvement":
            aq_fxn = probability_of_improvement
        elif aq_fxn == "expected_improvement":
            aq_fxn = expected_improvement
        elif aq_fxn == "upper_confidence_bound":
            aq_fxn = partial(upper_confidence_bound, _lambda=_lambda)
        elif aq_fxn == "greedy":
            aq_fxn = greedy
        else:
            raise ValueError(f"Unknown acquisition function: {aq_fxn}")

        results = self.predict(possible_x)
        # Needs to output predictions and probabilities



def prepare_data(inFile, outFile):
    reg_df = pd.read_csv(inFile)
    with open(outFile, "w") as f:
        for e in range(len(reg_df)):
            f.write(f'{{"prompt": "{reg_df.iloc[e]["prompt"]}", "completion": "{reg_df.iloc[e]["completion"]}"}}\n')

def split_data(inFile, outFile, n):
    with open(inFile, "r") as f:
        with open(outFile, "w") as out:
            for line in f.readlines()[0:n]:
                out.write(line)

if __name__ == "__main__":
    openai.api_key = "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"
    out_path = "out"
    if not os.path.exists(f"{out_path}"):
        os.makedirs(f"{out_path}")
    
    prepare_data(f"{out_path}/regression_dataset_sial_ratio.csv", f"{out_path}/tst_file.jsonl")
    
    split_data(f"{out_path}/tst_file.jsonl", f"{out_path}/tst_file_01.jsonl", 5)
    ft = AskTellFinetuning(f"{out_path}/tst_file_01.jsonl", "ada")
    # fineTuned_response = ft.fineTune()
    fineTuned_response = openai.FineTune.retrieve(id="ft-cJWERNpqVpEixitITjFKSjIP")
    with open(f"{out_path}/out01.dat", "w") as out:
        out.write(json.dumps(fineTuned_response, indent=4))

    # ask/tell procedure

    split_data(f"{out_path}/tst_file.jsonl", f"{out_path}/tst_file_02.jsonl", 8)
    ft = AskTellFinetuning(f"{out_path}/tst_file_02.jsonl", fineTuned_response["fine_tuned_model"])
    fineTuned_response = ft.fineTune()
    with open(f"{out_path}/out.dat02", "w") as out:
        out.write(json.dumps(fineTuned_response, indent=4))

    # print(openai.Model.list())