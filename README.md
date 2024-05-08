# Bayesian active learning with EPIG data acquisition

This repo contains code for two papers:


### [Prediction-oriented Bayesian active learning (AISTATS 2023)](https://arxiv.org/abs/2304.08151)

*Freddie Bickford Smith\*, Andreas Kirsch\*, Sebastian Farquhar, Yarin Gal, Adam Foster, Tom Rainforth*

Information-theoretic approaches to active learning have traditionally focused on maximising the information gathered about the model parameters, most commonly by optimising the BALD score.
We highlight that this can be suboptimal from the perspective of predictive performance.
For example, BALD lacks a notion of an input distribution and so is prone to prioritise data of limited relevance.
To address this we propose the expected predictive information gain (EPIG), an acquisition function that measures information gain in the space of predictions rather than parameters.
We find that using EPIG leads to stronger predictive performance compared with BALD across a range of datasets and models, and thus provides an appealing drop-in replacement.


### [Making better use of unlabelled data in Bayesian active learning (AISTATS 2024)](https://arxiv.org/abs/2404.17249)

*Freddie Bickford Smith, Adam Foster, Tom Rainforth*

Fully supervised models are predominant in Bayesian active learning.
We argue that their neglect of the information present in unlabelled data harms not just predictive performance but also decisions about what data to acquire.
Our proposed solution is a simple framework for semi-supervised Bayesian active learning.
We find it produces better-performing models than either conventional Bayesian active learning or semi-supervised learning with randomly acquired data.
It is also easier to scale up than the conventional approach.
As well as supporting a shift towards semi-supervised models, our findings highlight the importance of studying models and acquisition methods in conjunction.


## Getting set up

Clone the repo and move into it:

```bash
git clone https://github.com/fbickfordsmith/epig.git && cd epig
```

If you're not using a CUDA device, remove the `cudatoolkit` and `pytorch-cuda` dependencies in `environment.yaml`.

Create an environment using [Mamba](https://mamba.readthedocs.io) (or [Conda](https://conda.io), replacing `mamba` with `conda` below) and activate it:

```bash
mamba env create --file environment.yaml && mamba activate epig
```


## Reproducing the results

Run active learning with the default config:

```bash
python main.py
```

See [`jobs/`](/jobs/) for the commands used to run the active-learning experiments in the papers.

Each of the semi-supervised models we use in the AISTATS 2024 paper comprises an encoder and a prediction head.
Because we use fixed, deterministic encoders, we can compute the encoders' embeddings of all our inputs once up front and then save them to storage.
These embeddings just need to be moved into `data/` within this repo, and can be obtained from [`msn-embeddings`](https://github.com/fbickfordsmith/msn-embeddings.git), [`simclr-embeddings`](https://github.com/fbickfordsmith/simclr-embeddings.git), [`ssl-embeddings`](https://github.com/fbickfordsmith/ssl-embeddings.git) and [`vae-embeddings`](https://github.com/fbickfordsmith/vae-embeddings.git).


## Getting in touch

Contact [Freddie](https://github.com/fbickfordsmith) if you have any questions about this research or encounter any problems using the code.
This repo is a partial release of a bigger internal repo, and it's possible that errors were introduced when preparing this repo for release.


## Contributors

[Andreas Kirsch](https://github.com/BlackHC) wrote the original versions of the BALD and EPIG functions in this repo, along with the dropout layers, and advised on the code in general.
[Adam Foster](https://github.com/ae-foster) and [Joost van Amersfoort](https://github.com/y0ast) advised on the Gaussian-process implementation.
[Jannik Kossen](https://github.com/jlko) provided a repo template and advised on the code in general.

Credit for the unsupervised encoders we use in our semi-supervised models goes to the authors of [`disentangling-vae`](https://github.com/YannDubs/disentangling-vae), [`lightly`](https://github.com/lightly-ai/lightly), [`msn`](https://github.com/facebookresearch/msn) and [`solo-learn`](https://github.com/vturrisi/solo-learn), as well as the designers of the pretraining methods used.


## Citing this work

```bibtex
@article{bickfordsmith2023prediction,
    author = {{Bickford Smith}, Freddie and Kirsch, Andreas and Farquhar, Sebastian and Gal, Yarin and Foster, Adam and Rainforth, Tom},
    year = {2023},
    title = {Prediction-oriented {Bayesian} active learning},
    journal = {International Conference on Artificial Intelligence and Statistics},
}
```

```bibtex
@article{bickfordsmith2024making,
    author = {{Bickford Smith}, Freddie and Foster, Adam and Rainforth, Tom},
    year = {2024},
    title = {Making better use of unlabelled data in {Bayesian} active learning},
    journal = {International Conference on Artificial Intelligence and Statistics},
}
```
