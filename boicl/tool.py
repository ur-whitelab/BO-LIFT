from langchain.tools import BaseTool
from .asktell import AskTellFewShotTopk
from .pool import Pool
from typing import *
import os
import pandas as pd
from pydantic import BaseModel

from pydantic import BaseModel


class BOICLTool(BaseTool):
    name: str = "Experiment Designer"
    description: str = (
        "Propose or predict experiments using stateful ask-and-tell Bayes Optimizer. "
        "Syntax: Tell {{CSV_FILE}}. Adds training examples to model, {{CSV_FILE}}. No header and only two columns: x in column 0, y in column 1. "
        "Ask. Returns optimal experiment to run next. Must call Tell first. "
        "Best. Returns predicted experiment. Must call Tell first."
    )
    asktell: AskTellFewShotTopk | None = None
    pool: Pool | None = None

    def __init__(
        self,
        pool: Pool,
        asktell: Optional[AskTellFewShotTopk] = None,
    ):
        # call the parent class constructor
        super(BOICLTool, self).__init__()

        if asktell is None:
            asktell = AskTellFewShotTopk()
        self.asktell = asktell
        self.pool = pool

    def _run(self, query: str) -> str:
        if query.startswith("Ask"):
            cmd = "ask"
        elif query.startswith("Tell"):
            cmd = "tell"
            arg = query[4:].strip()
        elif query.startswith("Best"):
            cmd = "best"
        else:
            return "Invalid command to this tool"
        if cmd == "ask":
            results = self.asktell.ask(self.pool)
            return f"Optimal experiment to run next: {results[0][0]}"
        elif cmd == "tell":
            # check the path exists
            if not os.path.exists(arg):
                return f"File {arg} does not exist."
            # load the data column 0 is x, column 1 is y
            try:
                data = pd.read_csv(arg, header=None)
                for i in range(len(data)):
                    self.asktell.tell(data.iloc[i, 0], data.iloc[i, 1])
            except:
                return "Error in input file. Make sure that the file has no header and only has 2 columns: x and y. Remove all non-numeric values from y."
            return f"Added {len(data)} training examples."
        elif cmd == "best":
            results = self.asktell.ask(self.pool, aq_fxn="greedy")
            return f"Best experiment is: {results[0][0]}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
