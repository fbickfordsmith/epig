import hydra
import logging
import numpy as np
import os
import torch
from gpytorch.likelihoods import BernoulliLikelihood, SoftmaxLikelihood
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from pathlib import Path
from sklearn.cluster import kmeans_plusplus
from src.trainers.neural_network import NeuralNetworkTrainer
from src.utils import Dictionary, get_repo_status, format_time
from time import time


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    rng = call(cfg.rng)
    device = "cuda" if torch.cuda.is_available() and cfg.cuda else "cpu"

    start_time = time()
    is_first_al_step = True

    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)  # None if not running in Slurm

    logging.info(f"Seed: {cfg.rng.seed}")
    logging.info(f"Device: {device}")
    logging.info(f"Slurm job ID: {slurm_job_id}")

    # ----------------------------------------------------------------------------------------------
    logging.info(f"Making results dirs at {cfg.directories.results_run}")

    results_dir = Path(cfg.directories.results_run)
    results_subdirs = ["data_indices", "git"]

    if cfg.model_type in {"gp", "nn"}:
        results_subdirs += ["models", "training"]

    for subdir in results_subdirs:
        (results_dir / subdir).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------------------------------
    logging.info("Saving repo status")

    for filename, content in get_repo_status().items():
        with open(results_dir / "git" / filename, mode="wb") as _file:
            _file.write(content)

    # ----------------------------------------------------------------------------------------------
    logging.info("Loading data")

    data = instantiate(cfg.data, rng=rng)
    data.torch()
    data.to(device)

    for subset, inds in data.inds.items():
        np.savetxt(results_dir / "data_indices" / f"{subset}.txt", inds, fmt="%d")

    logging.info(f"Number of classes: {data.n_classes}")

    # ----------------------------------------------------------------------------------------------
    logging.info("Starting active learning")

    test_log = Dictionary()

    while True:
        n_labels_str = f"{data.n_train_labels:04}_labels"
        is_last_al_step = data.n_train_labels >= cfg.acquisition.n_train_labels_end

        logging.info(f"Number of labels: {data.n_train_labels}")

        # ------------------------------------------------------------------------------------------
        logging.info("Setting up trainer")

        if cfg.model_type == "gp":
            if cfg.likelihood_fn == "bernoulli":
                output_size = 1
                likelihood_fn = BernoulliLikelihood()
            else:
                output_size = data.n_classes
                likelihood_fn = SoftmaxLikelihood(num_classes=data.n_classes, mixing_weights=False)

            if cfg.init_length_scale_pdist:
                pool_inputs = data.main_dataset.data[data.inds["pool"]]
                mean_pool_pdist = torch.mean(torch.pdist(pool_inputs))
                trainer_kwargs = dict(init_length_scale=mean_pool_pdist)
            else:
                trainer_kwargs = {}

            train_inputs = data.main_dataset.data[data.inds["train"]]
            model = instantiate(cfg.model, inputs=train_inputs, output_size=output_size)
            model = model.to(device)
            likelihood_fn = likelihood_fn.to(device)
            trainer = instantiate(
                cfg.trainer, model=model, likelihood_fn=likelihood_fn, **trainer_kwargs
            )

        elif cfg.model_type == "nn":
            model = instantiate(cfg.model, input_shape=data.input_shape, output_size=data.n_classes)
            model = model.to(device)
            trainer = instantiate(cfg.trainer, model=model)

        elif cfg.model_type == "rf":
            model = instantiate(cfg.model)
            trainer = instantiate(cfg.trainer, model=model)

        else:
            raise ValueError

        if data.n_train_labels > 0:
            # --------------------------------------------------------------------------------------
            logging.info("Training")

            train_log = trainer.train(
                train_loader=data.get_loader("train"),
                val_loader=data.get_loader("val"),
            )

            if train_log != None:
                # ----------------------------------------------------------------------------------
                logging.info("Saving training log")

                formatting = {
                    "step": "{:05}".format,
                    "time": format_time,
                    "train_acc": "{:.4f}".format,
                    "train_loss": "{:.4f}".format,
                    "val_acc": "{:.4f}".format,
                    "val_loss": "{:.4f}".format,
                }

                train_log.save_to_csv(results_dir / "training" / f"{n_labels_str}.csv", formatting)
                np.savetxt(results_dir / "data_indices" / "train.txt", data.inds["train"], fmt="%d")

        is_in_save_steps = data.n_train_labels in cfg.model_save_steps
        model_dir_exists = (results_dir / "models").exists()

        if (is_first_al_step or is_last_al_step or is_in_save_steps) and model_dir_exists:
            # --------------------------------------------------------------------------------------
            logging.info("Saving model state")

            model_state = trainer.model.state_dict()
            torch.save(model_state, results_dir / "models" / f"{n_labels_str}.pth")

        # ------------------------------------------------------------------------------------------
        logging.info("Testing")

        with torch.inference_mode():
            test_acc, test_loss = trainer.test(data.get_loader("test"))

        test_log_update = {
            "n_labels": data.n_train_labels,
            "test_acc": test_acc.item(),
            "test_loss": test_loss.item(),
        }
        test_log.append(test_log_update)

        formatting = {
            "n_labels": "{:04}".format,
            "test_acc": "{:.4f}".format,
            "test_loss": "{:.4f}".format,
        }
        test_log.save_to_csv(results_dir / "testing.csv", formatting)

        if is_last_al_step:
            # --------------------------------------------------------------------------------------
            logging.info("Stopping active learning")

            break

        if cfg.acquisition.objective == "random":
            # --------------------------------------------------------------------------------------
            logging.info("Sampling data indices")

            acquired_pool_inds = rng.choice(
                len(data.inds["pool"]), size=cfg.acquisition.batch_size, replace=False
            )

        elif cfg.acquisition.objective == "badge":
            # --------------------------------------------------------------------------------------
            logging.info("Computing embeddings")

            assert isinstance(trainer, NeuralNetworkTrainer)

            for name, _ in trainer.model.named_parameters():
                if "weight" in name:
                    last_weight_name = name

            embedding_params = [last_weight_name]

            embeddings = trainer.compute_badge_embeddings(data.get_loader("pool"), embedding_params)

            _, acquired_pool_inds = kmeans_plusplus(
                embeddings.numpy(), cfg.acquisition.batch_size, random_state=rng.choice(int(1e6))
            )

        else:
            # --------------------------------------------------------------------------------------
            logging.info("Estimating uncertainty")

            if "TwoBells" in cfg.data.dataset._target_:
                if cfg.data.dataset.shift:
                    test_dist = data.test_dataset.input_dist
                else:
                    test_dist = data.main_dataset.input_dist

                target_inputs = test_dist.rvs(cfg.acquisition.n_target_samples, random_state=rng)
                target_inputs = torch.tensor(target_inputs, dtype=torch.float32)
                target_inputs = target_inputs.to(device)

            else:
                target_loader = data.get_loader("target")
                target_inputs, _ = next(iter(target_loader))

            with torch.inference_mode():
                scores = trainer.estimate_uncertainty(
                    pool_loader=data.get_loader("pool"),
                    target_inputs=target_inputs,
                    mode=cfg.acquisition.objective,
                    rng=rng,
                    epig_probs_target=cfg.acquisition.epig_probs_target,
                    epig_probs_adjustment=cfg.acquisition.epig_probs_adjustment,
                    epig_using_matmul=cfg.acquisition.epig_using_matmul,
                )

            scores = scores.numpy()
            scores = scores[cfg.acquisition.objective]

            acquired_pool_inds = np.argmax(scores)

        # ------------------------------------------------------------------------------------------
        logging.info(f"Acquiring with {cfg.acquisition.objective}")

        data.move_from_pool_to_train(acquired_pool_inds)

        is_first_al_step = False

    # ----------------------------------------------------------------------------------------------
    logging.info("Saving run time")

    run_time = format_time(time() - start_time)
    np.savetxt(results_dir / "run_time.txt", [run_time], fmt="%s")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Produce a complete stack trace
    main()
