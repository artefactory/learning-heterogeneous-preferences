<div align="center">

# Learning multiple multicriteria additive models from heterogeneous preferences


Vincent Auriau, Khaled Belahcene, Emmanuel Malherbe, Vincent Mousseau <br>
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

## Syntetic Experiments

## Real-World Experiments

## Using the model on you own data
<img align="right" width="200" src="doc/icon.png" />

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