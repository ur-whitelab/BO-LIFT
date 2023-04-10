# 🤖 BO-LIFT: Bayesian Optimization using in-context learning


![version](https://img.shields.io/badge/version-0.0.1-brightgreen)
[![paper](https://img.shields.io/badge/paper-arXiv-red)](#)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)


BO-LIFT does regression with uncertainties using frozen Large Language Models by using token probabilities.
It uses LangChain to select examples to create in-context learning prompts from training data.
By selecting examples, it can consider more training data than it fits in the model's context window.
Being able to predict uncertainty, allow the employment of interesting techniques such as Bayesian Optimization.


## Usage 💻

You need to set up your OpenAI API key in order to use BO-LIFT.
You can do that using the `os` Python library:

```
import os
os.environ["OPENAI_API_KEY"] = "<your-key-here>"
```

`bolift` provides different models depending on the prompt you want to use.
One example of usage can be seen in the following:

```
import bolift
asktell = bolift.AskTellFewShotTopk(
  ...args
)
```
Examples can be shown to the model by simply using the `tell` method:

```
for i in range(n):
  asktell.tell(<x_data>, <y_data>)
```

`bolift` will use these points to create the prompt for a prediction.
A prediction can be done using:

```
asktell.predict(<x_data>)
```
