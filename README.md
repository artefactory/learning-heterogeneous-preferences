<div align="center">

# Learning multiple multicriteria additive models from heterogeneous preferences


Vincent Auriau<sup>1, 2</sup>, Khaled Belahcène<sup>1</sup>, Emmanuel Malherbe<sup>2</sup>, Vincent Mousseau<sup>1</sup> <br>
<sup>1</sup> *MICS* - CentraleSupélec, <sup>2</sup> Artefact Research Center <br>

In ADT 2024. <br>
[[Paper]]()  [[Oral Presentation]]()<br>

</div>

<p align="center"><img width="95%" src="doc/PLS-3.png" /></p>

> **Abstract:** *Additive preference representation is standard in Multiple Criteria Decision Analysis, and learning such a preference model dates back from the UTA method. In this seminal work, an additive piece-wise linear model is inferred from a learning set composed of pairwise comparisons. In this setting, the learning set is provided by a single Decision-Maker (DM), and an additive model is inferred to match the learning set. We extend this framework to the case where (i) multiple DMs with heterogeneous preferences provide part of the learning set, and (ii) the learning set is provided as a whole without knowing which DM expressed each pairwise comparison. Hence, the problem amounts to inferring a preference model for each DM and simultaneously ``discovering'' the segmentation of the learning set. In this paper, we show that this problem is computationally difficult. We propose a mathematical programming based resolution approach to solve this Preference Learning and Segmentation problem (PLS). We also propose a heuristic to deal with large datasets. We study the performance of both algorithms through experiments using synthetic and real data.*

## Installation
Clone this repository:

```bash
git clone https://github.com/artefactory/learning-heterogeneous-preferences.git
```

Install the dependencies:
```bash
cd learning-heterogeneous-preferences
conda env create -f env.yml
```

## Synthetic Experiments

In order to run the experiments with synthetic data you can use the following command:

```bash
python run_synthetic_experiments save_xps --repetitions 4 --n_clusters 2 3 4 \
--n_criteria 6 --learning_set_size 128 1024 --error 0 5
```

It will save results in `save_xps` for four different datasets with all the combinations of parameters: 
- n_clusters = [1, 2, 3]
- n_criteria=6
- learning_set_size=[128, 1024]
- error=[0, 5]

## Real-World Experiments

The stated preferences for car dataset used in the paper can be downloaded [here](https://github.com/artefactory/choice-learn/blob/main/choice_learn/datasets/data/car.csv.gz).
It is also part of the [choice-learn](https://pypi.org/project/choice-learn/) package that can be installed with `pip install choice-learn`.


## Using the model on you own data
<img align="right" width="200" src="doc/icon.png" />

The different models can be used on your own data as follows:

```python
from python.models import UTA, ClusterUTA
from python.heuristics import Heuristic

model = model(**params)
model.fit(X, Y)

print(model.predict_utilitie(X))
````

All the models have lookalike signatures, in particular, in ```.fit(X, Y)```, X and Y must be:

More details are given in the docstrings of the models if you want to better understand the different hyper-parameters.

## License
This work is under the [MIT](./LICENSE) license.

## Citation
If you find this work useful for your research, please cite our paper:

```
@inproceedings{AuriauPLS:2024,
title={Learning multiple multicriteria additive models from heterogeneous preferences},
author={Auriau, Vincent and Belahcène, Khaled and Malherbe, Emmanuel and Mousseau, Vincent},
booktitle={Algorithmic Decision Theory},
year={2024},
}
```


<p align="center">
  <a href="https://www.artefact.com/data-consulting-transformation/artefact-research-center/">
    <img src="./doc/logo_arc.png" height="60" />
  </a>
  &emsp;
  &emsp;
  <a href="https://mics.centralesupelec.fr/">
    <img src="./doc/logo_CS.png" height="65" />
  </a>
  &emsp;
  &emsp;
  <a href="https://www.universite-paris-saclay.fr/">
    <img src="./doc/logo_paris_saclay.png" height="65" />
  </a>
</p>