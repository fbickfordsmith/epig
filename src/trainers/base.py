"""
B = batch size
N = number of examples
"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.logging import Dictionary


class Trainer:
    use_val_data: bool = True

    def test(self, loader: DataLoader, n_classes: int | None = None) -> dict:
        """
        loss = 1/N ∑_{i=1}^N L(x_i,y_i)

        Here we use
            L_1(x_i,y_i) = bin_loss(x_i,y_i) = 1[argmax p(y|x_i) == y_i]
            L_2(x_i,y_i) = nll_loss(x_i,y_i) = -log p(y_i|x_i).

        For stochastic models we use
            p(y|x_i)  = E_{p(θ)}[p(y|x_i,θ)]
                     ~= 1/K ∑_{j=1}^K p(y|x_i,θ_j), θ_j ~ p(θ).
        """
        self.eval_mode()

        test_log = Dictionary()
        n_examples = 0

        for inputs, labels in loader:
            if n_classes is not None:
                test_log_update = self.evaluate_test(inputs, labels, n_classes)
            else:
                test_log_update = self.evaluate_test(inputs, labels)

            test_log.append(test_log_update)

            n_examples += len(inputs)

        test_log = test_log.concatenate()

        for metric, scores in test_log.items():
            test_log[metric] = torch.sum(scores).item() / n_examples

        if "n_correct" in test_log:
            test_log["acc"] = test_log.pop("n_correct")

        return test_log


class DeterministicTrainer(Trainer):
    """
    Base trainer for a deterministic model.
    """

    def estimate_uncertainty(self, loader: DataLoader, method: str, seed: int) -> Tensor:
        self.eval_mode()

        scores = []

        for inputs_i, _ in loader:
            self.set_rng_seed(seed)
            scores_i = self.estimate_uncertainty_batch(inputs_i, method)  # [B,]
            scores += [scores_i.cpu()]

        return torch.cat(scores)  # [N,]


class StochasticTrainer(Trainer):
    """
    Base trainer for a stochastic model.
    """

    def estimate_uncertainty(
        self, loader: DataLoader, method: str, seed: int, inputs_targ: Tensor | None = None
    ) -> Tensor:
        self.eval_mode()

        use_epig_with_target_class_dist = (
            (method == "epig")
            and hasattr(self.epig_cfg, "target_class_dist")
            and self.epig_cfg.target_class_dist is not None
        )

        if use_epig_with_target_class_dist:
            scores = self.estimate_epig_using_target_class_dist(
                loader, n_input_samples=len(inputs_targ)
            )  # [N,]

        else:
            scores = []

            for inputs_i, _ in loader:
                self.set_rng_seed(seed)

                if method == "epig":
                    scores_i = self.estimate_epig_batch(inputs_i, inputs_targ)  # [B,]
                else:
                    scores_i = self.estimate_uncertainty_batch(inputs_i, method)  # [B,]

                scores += [scores_i.cpu()]

            scores = torch.cat(scores)  # [N,]

        return scores  # [N,]


class DifferentiableStochasticTrainer(StochasticTrainer):
    """
    Base trainer for a differentiable, stochastic model.
    """

    pass
