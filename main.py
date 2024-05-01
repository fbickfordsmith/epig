import logging
import os
from copy import deepcopy
from datetime import timedelta
from functools import partial
from pathlib import Path
from time import time
from typing import List

import hydra
import numpy as np
import torch
import wandb
from gpytorch.likelihoods import BernoulliLikelihood, SoftmaxLikelihood
from hydra.utils import call, instantiate
from laplace import ParametricLaplace
from numpy.random import Generator
from omegaconf import DictConfig
from sklearn.cluster import kmeans_plusplus
from torch.distributions import Gumbel
from torch.utils.data import DataLoader, TensorDataset

from src.coresets import (
    acquire_using_greedy_k_centers,
    acquire_using_k_means,
    acquire_using_k_means_plusplus,
    acquire_using_probcover,
    acquire_using_typiclust,
)
from src.data.active_learning import ActiveLearningData
from src.device import get_device
from src.logging import (
    Dictionary,
    get_formatters,
    prepend_to_keys,
    save_repo_status,
    save_run_to_wandb,
    set_up_wandb,
)
from src.trainers.base import DeterministicTrainer, Trainer
from src.trainers.gpytorch import GPyTorchTrainer
from src.trainers.pytorch import PyTorchTrainer


def get_gpytorch_trainer(
    data: ActiveLearningData, cfg: DictConfig, rng: Generator, device: str
) -> GPyTorchTrainer:
    if data.main_dataset.n_classes == 2:
        output_size = 1
        likelihood_fn = BernoulliLikelihood()
    else:
        output_size = data.main_dataset.n_classes
        likelihood_fn = SoftmaxLikelihood(num_classes=output_size, mixing_weights=False)

    train_inputs = data.main_dataset.data[data.main_inds["train"]]

    model = instantiate(cfg.model, inputs=train_inputs, output_size=output_size)
    model = model.to(device)

    seed = rng.choice(int(1e6))
    torch_rng = torch.Generator(device).manual_seed(seed)

    if cfg.init_length_scale_pdist:
        pool_inputs = data.main_dataset.data[data.main_inds["pool"]]
        mean_pool_pdist = torch.mean(torch.pdist(pool_inputs))
        trainer_kwargs = dict(init_length_scale=mean_pool_pdist)
    else:
        trainer_kwargs = {}

    trainer = instantiate(
        cfg.trainer, model=model, likelihood_fn=likelihood_fn, torch_rng=torch_rng, **trainer_kwargs
    )

    return trainer


def get_pytorch_trainer(
    data: ActiveLearningData, cfg: DictConfig, rng: Generator, device: str
) -> PyTorchTrainer:
    input_shape = data.main_dataset.input_shape
    output_size = data.main_dataset.n_classes

    model = instantiate(cfg.model, input_shape=input_shape, output_size=output_size)
    model = model.to(device)

    seed = rng.choice(int(1e6))
    torch_rng = torch.Generator(device).manual_seed(seed)

    return instantiate(cfg.trainer, model=model, torch_rng=torch_rng)


def get_sklearn_trainer(cfg: DictConfig) -> Trainer:
    model = instantiate(cfg.model)
    return instantiate(cfg.trainer, model=model)


def acquire_using_random(data: ActiveLearningData, cfg: DictConfig, rng: Generator) -> List[int]:
    n_pool = len(data.main_inds["pool"])
    return rng.choice(n_pool, size=cfg.acquisition.batch_size, replace=False).tolist()


def acquire_using_balanced_random(
    data: ActiveLearningData, cfg: DictConfig, rng: Generator, trainer: Trainer
) -> List[int]:
    """
    Randomly sample inputs, balanced by class. Whereas balanced_random uses the class labels of all
    the pool inputs and so cannot be used in practice, approx_balanced_random uses the model's
    predictions of the labels and so can be used in practice.
    """
    n_acquire = cfg.acquisition.batch_size // data.main_dataset.n_classes
    n_acquire = data.main_dataset.n_classes * [n_acquire]
    n_left = cfg.acquisition.batch_size % data.main_dataset.n_classes

    if n_left != 0:
        for _class in rng.choice(data.main_dataset.n_classes, size=n_left, replace=False):
            n_acquire[_class] += 1

    if cfg.acquisition.method == "approx_balanced_random":
        if isinstance(trainer, DeterministicTrainer):
            predict_fn = trainer.predict
        else:
            predict_fn = partial(trainer.marginal_predict, n_model_samples=trainer.n_samples_test)

        pool_labels = []

        for inputs, _ in data.get_loader("pool"):
            with torch.inference_mode():
                probs = predict_fn(inputs)

            pool_labels += [torch.argmax(probs, dim=-1).cpu()]

        pool_labels = torch.cat(pool_labels)
    else:
        pool_labels = data.main_dataset.targets[data.main_inds["pool"]]

    acquired_pool_inds = []

    for _class in range(data.main_dataset.n_classes):
        class_inds = torch.nonzero(pool_labels == _class).squeeze()
        n_sample = n_acquire[_class]
        acquired_pool_inds += rng.choice(class_inds, size=n_sample, replace=False).tolist()

    return acquired_pool_inds


def acquire_using_coreset_method(
    data: ActiveLearningData, cfg: DictConfig, rng: Generator, device: str
) -> List[int]:
    train_and_pool_inds = data.main_inds["train"] + data.main_inds["pool"]

    inputs = torch.clone(data.main_dataset.data[train_and_pool_inds])

    if hasattr(data.main_dataset, "data_mean") and hasattr(data.main_dataset, "data_std"):
        inputs *= data.main_dataset.data_std
        inputs += data.main_dataset.data_mean
        inputs /= torch.norm(inputs, dim=-1, keepdim=True)

    input_norms = torch.norm(inputs, dim=-1, keepdim=True)

    assert torch.allclose(input_norms, torch.ones_like(input_norms), atol=0.1)

    if cfg.acquisition.method == "probcover":
        inputs = inputs.to(device)
        graph = call(cfg.acquisition.probcover.graph_constructor, inputs=inputs)
        acq_method = partial(acquire_using_probcover, precomputed_graph=graph)
    else:
        acq_methods = {
            "greedy_k_centers": acquire_using_greedy_k_centers,
            "k_means": acquire_using_k_means,
            "k_means_plusplus": acquire_using_k_means_plusplus,
            "typiclust": acquire_using_typiclust,
        }
        acq_method = acq_methods[cfg.acquisition.method]

    # Here acquired_pool_inds indexes into inputs = concat(train_inputs, pool_inputs).
    acquired_pool_inds = acq_method(
        inputs,
        train_inds=list(range(data.n_train_labels)),
        n_acquire=cfg.acquisition.batch_size,
        rng=rng,
    )

    # Here acquired_pool_inds indexes into pool_inputs.
    acquired_pool_inds = np.array(acquired_pool_inds) - data.n_train_labels

    return acquired_pool_inds.tolist()


def acquire_using_badge_or_bait(
    data: ActiveLearningData, cfg: DictConfig, rng: Generator, device: str, trainer: Trainer
) -> List[int]:
    assert isinstance(trainer, PyTorchTrainer)

    if isinstance(trainer.model, ParametricLaplace):
        named_params = trainer.model.model.named_parameters()
    else:
        named_params = trainer.model.named_parameters()

    for name, param in named_params:
        if "weight" in name:
            last_weight_name = name
            last_weight_size = param.numel()

    embedding_params = [last_weight_name]

    if cfg.acquisition.method == "badge":
        # In addition to the memory cost of storing the embeddings, we need to consider that
        # kmeans_plusplus() has "total running time of O(nkd), when looking for a k-clustering
        # of n points in R^d" (https://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf).
        # 4GB memory budget -> ~4min for kmeans_plusplus(); 8GB memory budget -> ~7min.
        gb_per_embedding = last_weight_size * 4e-9  # 32-bit float = 4 bytes = 4e-9GB
    else:
        gb_per_embedding = data.main_dataset.n_classes * last_weight_size * 4e-9

    n_subsample = int(cfg.acquisition.badge_bait_memory_budget_gb / gb_per_embedding)

    if n_subsample < len(data.main_inds["pool"]):
        subsample = rng.choice(data.main_inds["pool"], size=n_subsample, replace=False)
        data.main_inds["pool_full"] = deepcopy(data.main_inds["pool"])
        data.main_inds["pool"] = subsample.tolist()

    if cfg.acquisition.method == "badge":
        if cfg.acquisition.badge_version == 1:
            embedding_fn = trainer.compute_badge_embeddings_v1
        else:
            embedding_fn = trainer.compute_badge_embeddings_v2

        embeddings = embedding_fn(data.get_loader("pool"), embedding_params)

        _, acquired_pool_inds = kmeans_plusplus(
            embeddings.numpy(),
            n_clusters=cfg.acquisition.batch_size,
            random_state=rng.choice(int(1e6)),
        )
    else:
        n_train = len(data.main_inds["train"])
        n_pool = len(data.main_inds["pool"])

        train_and_pool_inds = data.main_inds["train"] + data.main_inds["pool"]

        inputs = data.main_dataset.data[train_and_pool_inds].to(device)
        labels = data.main_dataset.targets[train_and_pool_inds].to(device)

        batch_size = data.batch_sizes["pool"]

        loader = DataLoader(
            dataset=TensorDataset(inputs, labels),
            batch_size=(len(inputs) if batch_size == -1 else batch_size),
            shuffle=False,
            drop_last=False,
            **data.loader_kwargs,
        )

        # Here acquired_pool_inds indexes into inputs = concat(train_inputs, pool_inputs).
        acquired_pool_inds = trainer.acquire_using_bait(
            loader,
            train_inds=range(n_train),
            pool_inds=range(n_train, n_train + n_pool),
            n_acquire=cfg.acquisition.batch_size,
            embedding_params=embedding_params,
        )

        # Here acquired_pool_inds indexes into pool_inputs.
        acquired_pool_inds = np.array(acquired_pool_inds) - n_train

    if "pool_full" in data.main_inds:
        remapped_acquired_pool_inds = []

        for acquired_pool_ind in acquired_pool_inds:
            ind = acquired_pool_ind  # Index into pool subsample
            ind = data.main_inds["pool"][ind]  # Index into data.main_dataset
            ind = np.flatnonzero(data.main_inds["pool_full"] == ind)  # Index into full pool
            remapped_acquired_pool_inds += ind.tolist()

        acquired_pool_inds = remapped_acquired_pool_inds
        data.main_inds["pool"] = data.main_inds.pop("pool_full")

    if isinstance(acquired_pool_inds, np.ndarray):
        acquired_pool_inds = acquired_pool_inds.tolist()

    return acquired_pool_inds


def acquire_using_uncertainty(
    data: ActiveLearningData, cfg: DictConfig, rng: Generator, device: str, trainer: Trainer
) -> List[int]:
    if "TwoBells" in cfg.data.dataset._target_:
        if cfg.data.dataset.shift:
            test_dist = data.test_dataset.input_dist
        else:
            test_dist = data.main_dataset.input_dist

        target_inputs = test_dist.rvs(cfg.acquisition.epig.n_target_samples, random_state=rng)
        target_inputs = torch.tensor(target_inputs, dtype=torch.float32, device=device)
    else:
        target_loader = data.get_loader("target")
        target_inputs, _ = next(iter(target_loader))

    acq_kwargs = dict(
        loader=data.get_loader("pool"), method=cfg.acquisition.method, seed=rng.choice(int(1e6))
    )

    if cfg.acquisition.method == "epig":
        acq_kwargs = dict(inputs_targ=target_inputs, **acq_kwargs)

    with torch.inference_mode():
        scores = trainer.estimate_uncertainty(**acq_kwargs)

    if cfg.acquisition.batch_size == 1:
        acquired_pool_inds = [torch.argmax(scores).item()]
    else:
        # Use stochastic batch acquisition (https://arxiv.org/abs/2106.12059).
        scores = torch.log(scores) + Gumbel(loc=0, scale=1).sample(scores.shape)
        acquired_pool_inds = torch.argsort(scores)[-cfg.acquisition.batch_size :]
        acquired_pool_inds = acquired_pool_inds.tolist()

    return acquired_pool_inds


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    device = get_device(cfg.use_gpu)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", default=None)  # None if not running in Slurm
    rng = call(cfg.rng)
    formatters = get_formatters()

    if cfg.use_gpu and (device not in {"cuda", "mps"}):
        logging.warning(f"Device: {device}")
    else:
        logging.info(f"Device: {device}")

    logging.info(f"Slurm job ID: {slurm_job_id}")
    logging.info(f"Seed: {cfg.rng.seed}")
    logging.info(f"Making results dirs at {cfg.directories.results_run}")

    results_dir = Path(cfg.directories.results_run)

    for subdir in cfg.directories.results_subdirs[cfg.model_type]:
        Path(subdir).mkdir(parents=True, exist_ok=True)

    save_repo_status(results_dir / "git")

    if cfg.wandb.use:
        set_up_wandb(cfg, slurm_job_id)

    # ----------------------------------------------------------------------------------------------
    logging.info("Loading data")

    data = instantiate(cfg.data, rng=rng, device=device)
    data.convert_datasets_to_torch()

    for subset, inds in data.main_inds.items():
        np.savetxt(results_dir / "data_indices" / f"{subset}.txt", inds, fmt="%d")

    np.savetxt(results_dir / "data_indices" / "test.txt", data.test_inds, fmt="%d")

    logging.info(f"Number of classes: {data.main_dataset.n_classes}")

    # ----------------------------------------------------------------------------------------------
    logging.info("Starting active learning")

    is_first_al_step = True
    start_time = time()
    test_log = Dictionary()

    while True:
        n_labels_str = f"{data.n_train_labels:04}_labels"
        is_last_al_step = data.n_train_labels >= cfg.acquisition.n_train_labels_end

        # ------------------------------------------------------------------------------------------
        logging.info(f"Number of labels: {data.n_train_labels}")
        logging.info("Setting up trainer")

        if cfg.model_type == "gpytorch":
            trainer = get_gpytorch_trainer(data, cfg, rng, device)

        elif cfg.model_type == "pytorch":
            trainer = get_pytorch_trainer(data, cfg, rng, device)

        elif cfg.model_type == "sklearn":
            trainer = get_sklearn_trainer(cfg)

        else:
            raise ValueError

        if data.n_train_labels > 0:
            # --------------------------------------------------------------------------------------
            logging.info("Training")

            train_step, train_log = trainer.train(
                train_loader=data.get_loader("train"), val_loader=data.get_loader("val")
            )

            if train_step is not None:
                if train_step < cfg.trainer.n_optim_steps_max - 1:
                    logging.info(f"Training stopped early at step {train_step}")
                else:
                    logging.warning(f"Training stopped before convergence at step {train_step}")

            if train_log is not None:
                train_log.save_to_csv(results_dir / "training" / f"{n_labels_str}.csv", formatters)

            np.savetxt(
                results_dir / "data_indices" / "train.txt", data.main_inds["train"], fmt="%d"
            )

        is_in_save_steps = data.n_train_labels in cfg.model_save_steps
        model_dir_exists = (results_dir / "models").exists()

        if (is_first_al_step or is_last_al_step or is_in_save_steps) and model_dir_exists:
            logging.info("Saving model checkpoint")

            if isinstance(trainer.model, ParametricLaplace):
                model_state = trainer.model.model.state_dict()
            else:
                model_state = trainer.model.state_dict()

            torch.save(model_state, results_dir / "models" / f"{n_labels_str}.pth")

        # ------------------------------------------------------------------------------------------
        logging.info("Testing")

        if cfg.adjust_test_predictions:
            test_labels = data.test_dataset.targets[data.test_inds]
            test_kwargs = dict(n_classes=len(torch.unique(test_labels)))
        else:
            test_kwargs = {}

        with torch.inference_mode():
            test_metrics = trainer.test(data.get_loader("test"), **test_kwargs)

        test_metrics_str = ", ".join(
            f"{key} = {formatters[f'test_{key}'](value)}" for key, value in test_metrics.items()
        )

        logging.info(f"Test metrics: {test_metrics_str}")

        test_log.append({"n_labels": data.n_train_labels, **prepend_to_keys(test_metrics, "test")})
        test_log.save_to_csv(results_dir / "testing.csv", formatters)

        if cfg.wandb.use:
            wandb.log({key: values[-1] for key, values in test_log.items()})

        if is_last_al_step:
            logging.info("Stopping active learning")
            break

        # ------------------------------------------------------------------------------------------
        logging.info(
            f"Acquiring {cfg.acquisition.batch_size} label(s) using {cfg.acquisition.method}"
        )

        uncertainty_types = {"bald", "epig", "marg_entropy", "mean_std", "pred_margin", "var_ratio"}

        if cfg.acquisition.method == "random":
            acquired_pool_inds = acquire_using_random(data, cfg, rng)

        elif cfg.acquisition.method in {"approx_balanced_random", "balanced_random"}:
            acquired_pool_inds = acquire_using_balanced_random(data, cfg, rng, trainer)

        elif cfg.acquisition.method in {"greedy_k_centers", "k_means", "probcover", "typiclust"}:
            acquired_pool_inds = acquire_using_coreset_method(data, cfg, rng, device)

        elif cfg.acquisition.method in {"badge", "bait"}:
            acquired_pool_inds = acquire_using_badge_or_bait(data, cfg, rng, device, trainer)

        elif cfg.acquisition.method in uncertainty_types:
            acquired_pool_inds = acquire_using_uncertainty(data, cfg, rng, device, trainer)

        else:
            raise ValueError

        data.move_from_pool_to_train(acquired_pool_inds)
        is_first_al_step = False

    run_time = timedelta(seconds=(time() - start_time))
    np.savetxt(results_dir / "run_time.txt", [str(run_time)], fmt="%s")

    if cfg.wandb.use:
        save_run_to_wandb(results_dir, cfg.directories.results_subdirs[cfg.model_type])
        wandb.finish()  # Ensure each run in a Hydra multirun is logged separately


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Produce a complete stack trace
    main()
