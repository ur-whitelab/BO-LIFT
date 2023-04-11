# 🤖 BO-LIFT: Bayesian Optimization using in-context learning


![version](https://img.shields.io/badge/version-0.0.1-brightgreen)
[![paper](https://img.shields.io/badge/paper-arXiv-red)](#)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)


BO-LIFT does regression with uncertainties using frozen Large Language Models by using token probabilities.
It uses LangChain to select examples to create in-context learning prompts from training data.
By selecting examples, it can consider more training data than it fits in the model's context window.
Being able to predict uncertainty, allow the employment of interesting techniques such as Bayesian Optimization.

## Table of content
-[BO-LIFT](#-bo-lift-bayesian-optimization-using-in-context-learning)
  - [Install](#)
  - [Usage](#usage-)
    - [Creating the model](#creating-the-model)
    - [Making a prediction](#making-a-prediction)
    - [Improving the model](#improving-the-model)
    - [Inverse design](#inverse-design)
  - [Citation](#citation)


## Usage 💻

You need to set up your OpenAI API key in order to use BO-LIFT.
You can do that using the `os` Python library:

```py
import os
os.environ["OPENAI_API_KEY"] = "<your-key-here>"
```

### Creating the model

`bolift` provides different models depending on the prompt you want to use.
One example of usage can be seen in the following:

```py
import bolift
asktell = bolift.AskTellFewShotTopk(
  x_formatter=lambda x: f"iupac name {x}",
  y_name="measured log solubility in mols per litre",
  y_formatter=lambda y: f"{y:.2f}",
  model="gpt-4",
  selector_k=5,
  temperature=0.7,
)
```
Other arguments can be used to customize the prompt (`prefix`, `prompt_template`, `suffix`) and the in-context learning procedure (`use_quantiles`, `n_quantiles`).
Refer to the notebooks available in the paper directory to see examples on how to use bolift.

### Making a prediction

Examples can be shown to the model by simply using the `tell` method:

```py
asktell.tell("3-chloroaniline", -1.37)
asktell.tell("nitromethane", 0.26)
asktell.tell("1-bromo-2-methylpropane", -2.43)
asktell.tell("3-chlorophenol", -0.7)
```

`bolift` will use these points to create the prompt for a prediction. 
A prediction can be done using:

```py
yhat = asktell.predict("3-(3,4-dichlorophenyl)-1-methoxy-1-methylurea")
```

The prediction returns a probability distribution.
We can look into the predictions more easily by using the `mean` and the `std` methods:

```py
yhat.mean(), yhat.std()
```

`3-(3,4-dichlorophenyl)-1-methoxy-1-methylurea` has a LogS solubility of $-3.592$.
However, in this example `bolift` predicted a LogS of $-0.187 \pm 1.751$, which is not a good prediction.
The reason is that only 4 points were told to the model

### Improving the model

In the ask/tell interface, we can use Bayesian optimization and the `ask` method to select a new datapoint from a pool.
`bolift` also implements a pool object to efficiently deal with a pool of examples.

```py
pool_list = [
  "phenol",
  "1-methoxy-4-prop-1-enylbenzene",
  "2-aminophenol",
  "1,1-dimethyl-3-(8-tricyclo[5.2.1.02,6]decanyl)urea",
  "1,1,2,3,4,4-hexachlorobuta-1,3-diene"
]
pool = bolift.Pool(list(pool_list), formatter=lambda x: f"iupac name {x}")

asktell.ask(pool, aq_fxn="expected_improvement")
```

Notice that a formatter is also needed for the pool.
In this example, the expected `improvement_acquisition` function was used.
`upper_confidence_bound`, `greedy`, `probability_of_improvement` and `random` are also accepted values.
The `ask` method returns the selected elements (by default, it returns one element. The `k` argument can be used customize that), the value of the acquisition function and the prediction for that element.

Subsequently, we can address a label and `tell` this new points to the model.

### Inverse design

Aiming to propose new data, `bolift` implements another approach to generate data.
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

Please, cite [Ramos et al.](#):
```
@article{ramos2023bolift,
  title={Bayesian Optimization of Catalysts With In-context Learning},
  author={Ramos, Mayk and Michtavy, Shane and Porosoff, Marc and White, Andrew D},
  journal={arXiv preprint arXiv:submit/4836185},
  year={2023},
}
```
