| Code for "Making better use of unlabelled data in Bayesian active learning" (AISTATS 2024) will be added by 1 May 2024 |
| - |

# Prediction-oriented Bayesian active learning

Freddie Bickford Smith*, Andreas Kirsch*, Sebastian Farquhar, Yarin Gal, Adam Foster, Tom Rainforth \
International Conference on Artificial Intelligence and Statistics (AISTATS), 2023

[![Python](https://img.shields.io/badge/Python-3670A0.svg?style=flat-square&logo=Python&logoColor=ffdd54)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-black.svg?style=flat-square)](LICENSE.md)
[![arXiv](https://img.shields.io/badge/arXiv-2304.08151-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2304.08151)


## Abstract

Information-theoretic approaches to active learning have traditionally focused on maximising the information gathered about the model parameters, most commonly by optimising the BALD score.
We highlight that this can be suboptimal from the perspective of predictive performance.
For example, BALD lacks a notion of an input distribution and so is prone to prioritise data of limited relevance.
To address this we propose the expected predictive information gain (EPIG), an acquisition function that measures information gain in the space of predictions rather than parameters.
We find that using EPIG leads to stronger predictive performance compared with BALD across a range of datasets and models, and thus provides an appealing drop-in replacement.


## Running the code

Create a Conda environment:

```bash
conda env create --file environment.yaml
```

Run active learning with the default config:

```bash
python main.py
```

See the `jobs` directory for the commands used to run the experiments in the paper.


## Contact

Get in touch with [Freddie](https://github.com/fbickfordsmith) if you have any questions about this research or encounter any problems using the code.
This repo is a partial release of a bigger internal repo, and it's possible that errors were introduced in the process of preparing this repo for release.


## Contributors

[Andreas Kirsch](https://github.com/BlackHC) wrote the original versions of the BALD and EPIG functions in this repo, along with the dropout layers, and advised on the code in general.
[Adam Foster](https://github.com/ae-foster) and [Joost van Amersfoort](https://github.com/y0ast) advised on the Gaussian-process implementation.
[Jannik Kossen](https://github.com/jlko) provided a repo template and advised on the code in general.


## Citation

Please cite our paper if you use our code or ideas in your work.

```bibtex
@article{
    bickfordsmith2023prediction,
    author = {{Bickford Smith}, Freddie and Kirsch, Andreas and Farquhar, Sebastian and Gal, Yarin and Foster, Adam and Rainforth, Tom},
    year = {2023},
    title = {Prediction-oriented {Bayesian} active learning},
    journal = {International Conference on Artificial Intelligence and Statistics},
}
```
