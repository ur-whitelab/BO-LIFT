from langchain.tools import BaseTool
from .asktell import AskTellFewShotTopk
from .pool import Pool
from typing import *
import os
import pandas as pd


class BOLiftTool(BaseTool):
    name = "Experiment Designer"
    description = "Propose or predict experiments using stateful ask-and-tell. "
    "Syntax: Tell {CSV_FILE}. Adds training examples to model. "
    "Ask. Returns best experiment to run next. "
    asktell: AskTellFewShotTopk = None
    pool: Pool = None

    def __init__(self, pool: Pool, asktell: Optional[AskTellFewShotTopk] = None, ):
        # call the parent class constructor
        super(BOLiftTool, self).__init__()

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
        else:
            return "Invalid command. Syntax: Ask. Returns best experiment to run next. Tell {CSV_FILE}. Adds training examples to model."
        if cmd == "ask":
            results = self.asktell.ask(self.pool)
            return f"Best experiments to run next: {results[0][0]}"
        elif cmd == "tell":
            # check the path exists
            if not os.path.exists(arg):
                return f"File {arg} does not exist."
            # load the data column 0 is x, column 1 is y
            data = pd.read_csv(arg, header=None)
            for i in range(len(data)):
                self.asktell.tell(data.iloc[i, 0], data.iloc[i, 1])
            return f"Added {len(data)} training examples."
            
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
