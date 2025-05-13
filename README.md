# 🤖 BO-ICL: Bayesian Optimization using in-context learning

![version](https://img.shields.io/badge/version-0.0.1-brightgreen)
[![paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2304.05341)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

BO-ICL does regression with uncertainties using frozen Large Language Models by using token probabilities.
It uses LangChain to select examples to create in-context learning prompts from training data.
By selecting examples, it can consider more training data than it fits in the model's context window.
Being able to predict uncertainty, allow the employment of interesting techniques such as Bayesian Optimization.

## Table of content

- [BO-ICL](#-bo-icl-bayesian-optimization-using-in-context-learning)
  - [Install](#install-)
  - [Usage](#usage-)
    - [Quickstart](#quickstart-)
    - [Customising the model](#customising-the-model)
    - [Inverse design](#inverse-design)
  - [Citation](#citation)

## Install 📦

boicl can simply be installed using pip:

```bash
pip install boicl
```

Some additional requirements are needed to use the Gaussian Process Regressor (GPR) module.
They can also be installed using pip:

```bash
pip install boicl[gpr]
```

## Usage 💻

You need to set up your OpenAI API key in order to use BO-ICL.
You can do that using the `os` Python library:

```py
import os
os.environ["OPENAI_API_KEY"] = "<your-key-here>"
```

### Quickstart 🔥

`boicl` provides a simple interface to use the model.

```py
# Create the model object
asktell = boicl.AskTellFewShotTopk()

# Tell some points to the model
asktell.tell("1-bromopropane", -1.730)
asktell.tell("1-bromopentane", -3.080)
asktell.tell("1-bromooctane", -5.060)
asktell.tell("1-bromonaphthalene", -4.35)

# Make a prediction
yhat = asktell.predict("1-bromobutane")
print(yhat.mean(), yhat.std())
```

This prediction returns $-2.92 \pm 1.27$.

Further improvements can be done by using Bayesian Optimization.

```py
# Create a list of examples
pool_list = [
  "1-bromoheptane",
  "1-bromohexane",
  "1-bromo-2-methylpropane",
  "butan-1-ol"
]

# Create the pool object
pool=boicl.Pool(pool_list)

# Ask the next point
asktell.ask(pool)

# Output:
(['1-bromo-2-methylpropane'], [-1.284916344093158], [-1.92])

```

Where the first value is the selected point, the second value is the value of the acquisition function, and the third value is the predicted mean.

Let's tell this point to the model with its correct label and make a prediction:

```py
asktell.tell("1-bromo-2-methylpropane", -2.430)

yhat = asktell.predict("1-bromobutane")
print(yhat.mean(), yhat.std())
```

This prediction returns $-1.866 \pm 0.012$.
Which is closer to the label of -2.370 for the 1-bromobutane and the uncertainty also decreased.

### Customising the model

`boicl` provides different models depending on the prompt you want to use.
One example of usage can be seen in the following:

```py
import boicl
asktell = boicl.AskTellFewShotTopk(
  x_formatter=lambda x: f"iupac name {x}",
  y_name="measured log solubility in mols per litre",
  y_formatter=lambda y: f"{y:.2f}",
  model="gpt-4",
  selector_k=5,
  temperature=0.7,
)
```

Other arguments can be used to customize the prompt (`prefix`, `prompt_template`, `suffix`) and the in-context learning procedure (`use_quantiles`, `n_quantiles`).
Additionally, we implemented other models. A brief list can be seen below:

- AskTellFewShotTopk;
- AskTellFinetuning;
- AskTellRidgeKernelRegression;
- AskTellGPR;
- AskTellNearestNeighbor.

Refer to the [notebooks](https://github.com/ur-whitelab/BO-ICL/tree/main/paper) available in the paper directory to see examples of how to use boicl and the [paper](https://arxiv.org/abs/2304.05341) for a detailed description of the classes.

### Inverse design

Aiming to propose new data, `boicl` implements another approach to generate data.
After following a similar procedure to `tell` datapoints to the model, the `inv_predict` can be used to do an inverse prediction.
For carrying an inverse design out, we query the label we want and the model should generate a data that corresponds to that label:

```py
data_x = [
"A 15 wt% tungsten carbide catalyst was prepared with Fe dopant metal at 0.5 wt% and carburized at 835 °C. The reaction was run at 280 °C, resulting in a CO yield of",
"A 15 wt% tungsten carbide catalyst was prepared with Fe dopant metal at 0.5 wt% and carburized at 835 °C. The reaction was run at 350 °C, resulting in a CO yield of",
...
]

data_y = [
1.66,
3.03,
...
]


for i in range(n):
  asktell.tell(data_x[i], data_y[i]

asktell.inv_predict(20.0)
```

The data for that is available in the paper directory.
This generated the following procedure:

```
the synthesis procedure:"A 30 wt% tungsten carbide catalyst was prepared with Cu dopant metal at 5 wt% and carburized at 835 C. The reaction was run at 350 ºC"
```

### Citation

Please, cite [Ramos et al.](https://arxiv.org/abs/2304.05341):

```
@misc{ramos2023bayesian,
      title={Bayesian Optimization of Catalysts With In-context Learning},
      author={Mayk Caldas Ramos and Shane S. Michtavy and Marc D. Porosoff and Andrew D. White},
      year={2023},
      eprint={2304.05341},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph}
}
```
