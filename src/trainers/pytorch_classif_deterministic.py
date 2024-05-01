"""
B = batch size
Cl = number of classes
E = embedding size
F = number of features
N = number of examples
"""

import math
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.func import functional_call, grad, vmap
from torch.nn.functional import log_softmax, nll_loss
from torch.utils.data import DataLoader

from src.metrics import accuracy_from_marginals
from src.trainers.base_classif_logprobs import LogprobsClassificationDeterministicTrainer
from src.trainers.pytorch_classif import PyTorchClassificationTrainer
from src.typing import IndexSequence


class PyTorchClassificationDeterministicTrainer(
    PyTorchClassificationTrainer, LogprobsClassificationDeterministicTrainer
):
    def eval_mode(self) -> None:
        self.model.eval()

    def predict(self, inputs: Tensor) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, *F]

        Returns:
            Tensor[float], [N, Cl]
        """
        features = self.model(inputs)  # [N, Cl]
        return log_softmax(features, dim=-1)  # [N, Cl]

    def evaluate_train(self, inputs: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        loss = -1/N ∑_{i=1}^N log p(y_i|x_i)
        """
        logprobs = self.predict(inputs)  # [N, Cl]

        acc = accuracy_from_marginals(logprobs, labels)  # [1,]
        nll = nll_loss(logprobs, labels)  # [1,]

        return acc, nll  # [1,], [1,]

    def compute_badge_pseudoloss_v1(self, inputs: Tensor) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, *F]

        Returns:
            Tensor[float], [N,]
        """
        logprobs = self.predict(inputs)  # [N, Cl]
        pseudolabels = torch.argmax(logprobs, dim=-1)  # [N,]

        return nll_loss(logprobs, pseudolabels, reduction="none")  # [N,]

    def compute_badge_pseudoloss_v2(
        self, _input: Tensor, grad_params: dict, no_grad_params: dict
    ) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [1, *F]

        Returns:
            Tensor[float], [1,]
        """
        features = functional_call(
            self.model, (grad_params, no_grad_params), _input[None, :]
        )  # [1, Cl]

        logprobs = log_softmax(features, dim=-1)  # [1, Cl]
        pseudolabel = torch.argmax(logprobs, dim=-1)  # [1,]

        return nll_loss(logprobs, pseudolabel)  # [1,]

    def acquire_using_bait(
        self,
        loader: DataLoader,
        train_inds: IndexSequence,
        pool_inds: IndexSequence,
        n_acquire: int,
        embedding_params: Sequence[str],
        _lambda: float = 1.0,
        oversampling_multiplier: int = 2,
    ) -> List[int]:
        """
        We use compute_bait_embeddings_v2() instead of compute_bait_embeddings_v1() because it's
        much faster. The two versions are equivalent:
        >>> embeddings_i1 = self.compute_bait_embeddings_v1(inputs_i, embedding_params)
        >>> embeddings_i2 = self.compute_bait_embeddings_v2(inputs_i, embedding_params)
        >>> torch.allclose(embeddings_i1, embeddings_i2, atol=1e-6)  # -> True
        """
        n_train = len(train_inds)

        embeddings_pool = []
        fisher_pool = fisher_train = counter = 0

        for inputs_i, _ in loader:
            is_in_train = np.isin(range(counter, counter + len(inputs_i)), train_inds)  # [B,]
            counter += len(inputs_i)

            embeddings_i = self.compute_bait_embeddings_v2(inputs_i, embedding_params)  # [B, Cl, E]
            embeddings_pool += [embeddings_i[~is_in_train]]

            fishers_i = embeddings_i.transpose(1, 2) @ embeddings_i  # [B, E, E]
            fisher_pool += torch.sum(fishers_i[~is_in_train], dim=0)  # [E, E]
            fisher_train += torch.sum(fishers_i[is_in_train], dim=0)  # [E, E]

        embeddings_pool = torch.cat(embeddings_pool, dim=0)  # [N_p, Cl, E]
        embeddings_pool *= math.sqrt(n_acquire / (n_train + n_acquire))  # [N_p, Cl, E]

        fisher_all = (fisher_pool + fisher_train) / (len(train_inds) + len(pool_inds))
        fisher_selected = _lambda * torch.eye(
            embeddings_pool.shape[-1], device=fisher_train.device
        )  # [E, E]
        fisher_selected += fisher_train / (n_train + n_acquire)  # [E, E]
        fisher_selected_inverse = torch.inverse(fisher_selected)  # [E, E]

        selected_inds, fisher_selected_inverse = self.forward_select_for_bait(
            embeddings_pool,
            fisher_all,
            fisher_selected_inverse,
            oversampling_multiplier * n_acquire,
        )  # [oversampling_multiplier * N_a,], [E, E]

        selected_inds = self.backward_select_for_bait(
            embeddings_pool[selected_inds],
            fisher_all,
            fisher_selected_inverse,
            n_acquire,
            selected_inds,
        )  # [N_a,]

        selected_inds = [pool_inds[ind] for ind in selected_inds]  # [N_a,]

        return selected_inds  # [N_a,]

    def compute_bait_embeddings_v1(self, inputs: Tensor, embedding_params: Sequence[str]) -> Tensor:
        self.eval_mode()

        logprobs = self.predict(inputs)  # [N, Cl]

        embeddings = []

        for logprobs_i in logprobs:
            embedding_i = []

            for logprobs_ij in logprobs_i:
                # Prevent the grad attribute of each tensor accumulating a sum of gradients.
                self.model.zero_grad()

                logprobs_ij.backward(retain_graph=True)

                embedding_ij = []

                for name, param in self.model.named_parameters():
                    if name in embedding_params:
                        embedding_ij += [param.grad.flatten()]  # [E',]

                embedding_ij = torch.cat(embedding_ij)  # [E,]

                # Multiply by the square root of probs_ij. The square root is included because the
                # Fisher matrix is E_{p(y|x,θ)}[g g^T] where g = ∇_θ log p(y|x,θ) and we compute it
                # by a matrix multiplication that combines the outer product with the expectation.
                embedding_ij *= torch.exp(0.5 * logprobs_ij)  # [E,]

                embedding_i += [embedding_ij]  # [E,]

            embedding_i = torch.stack(embedding_i)  # [Cl, E]
            embeddings += [embedding_i]  # [Cl, E]

        # Detaching the embeddings from the computational graph is needed to avoid a memory leak.
        # See https://discuss.pytorch.org/t/memory-leak-debugging-and-common-causes/67339.
        embeddings = torch.stack(embeddings).detach()  # [N, Cl, E]

        return embeddings  # [N, Cl, E]

    def compute_bait_embeddings_v2(self, inputs: Tensor, embedding_params: Sequence[str]) -> Tensor:
        self.eval_mode()

        grad_params, no_grad_params = self.split_params(embedding_params)
        compute_grad = grad(self.compute_logprobs, argnums=2)
        compute_grad = vmap(compute_grad, in_dims=(0, None, None, None))

        with torch.inference_mode():
            logprobs = self.predict(inputs)  # [N, Cl]

        embeddings = []

        for _class in range(logprobs.shape[-1]):
            gradient_dict = compute_grad(inputs, _class, grad_params, no_grad_params)

            embeddings_j = []

            for name, gradient in gradient_dict.items():
                gradient = gradient.flatten(start_dim=1)  # [N, E']
                embeddings_j += [gradient]

            embeddings_j = torch.cat(embeddings_j, dim=-1)  # [N, E]
            embeddings += [embeddings_j]

        embeddings = torch.stack(embeddings, dim=1)  # [N, Cl, E]

        # Multiply by the square root of probs. The square root is included because the Fisher
        # matrix is E_{p(y|x,θ)}[g g^T] where g = ∇_θ log p(y|x,θ) and we compute it by a matrix
        # multiplication that combines the outer product with the expectation.
        embeddings *= torch.exp(0.5 * logprobs[:, :, None])  # [N, Cl, E]

        return embeddings  # [N, Cl, E]

    def compute_logprobs(
        self, _input: Tensor, _class: int, grad_params: dict, no_grad_params: dict
    ) -> Tensor:
        features = functional_call(
            self.model, (grad_params, no_grad_params), _input[None, :]
        )  # [1, Cl]

        logprobs = log_softmax(features, dim=-1)  # [1, Cl]

        return logprobs[0, _class]  # [1,]

    @staticmethod
    def forward_select_for_bait(
        embeddings_pool: Tensor,
        fisher_pool: Tensor,
        fisher_selected_inverse: Tensor,
        n_acquire: int,
    ) -> Tuple[List[int], Tensor]:
        _, n_classes, _ = embeddings_pool.shape

        I = torch.eye(n_classes, device=embeddings_pool.device)  # [Cl, Cl]
        M_inv = fisher_selected_inverse  # [E, E]
        V = embeddings_pool  # [N_p, Cl, E]

        selected_inds = []

        for _ in range(n_acquire):
            A_inv = torch.inverse(I + V @ M_inv @ V.transpose(1, 2))  # [N_p, Cl, Cl]

            inds_inf = torch.where(torch.isinf(A_inv))
            A_inv[inds_inf] = (
                torch.sign(A_inv[inds_inf]) * torch.finfo(V.dtype).max
            )  # [N_p, Cl, Cl]

            # [N_p, Cl, E] @ [E, E] @ [E, E] @ [E, E] @ [N_p, E, Cl] @ [Cl, Cl]
            matrix_prod = (
                V @ M_inv @ fisher_pool @ M_inv @ V.transpose(1, 2) @ A_inv
            )  # [N_p, Cl, Cl]

            trace = torch.vmap(torch.trace)(matrix_prod)  # [N_p,]

            for j in torch.argsort(trace, descending=True):
                if j.item() not in selected_inds:
                    selected_ind = j.item()
                    break

            A_inv = torch.inverse(I + V[selected_ind] @ M_inv @ V[selected_ind].T)  # [Cl, Cl]

            # [E, E] - [E, E] @ [E, Cl] @ [Cl, Cl] @ [Cl, E] @ [E, E]
            M_inv = M_inv - M_inv @ V[selected_ind].T @ A_inv @ V[selected_ind] @ M_inv  # [E, E]

            selected_inds += [selected_ind]

        return selected_inds, M_inv

    @staticmethod
    def backward_select_for_bait(
        embeddings_selected: Tensor,
        fisher_pool: Tensor,
        fisher_selected_inverse: Tensor,
        n_acquire: int,
        selected_inds: List[int],
    ) -> List[int]:
        _, n_classes, _ = embeddings_selected.shape

        I = torch.eye(n_classes, device=embeddings_selected.device)  # [Cl, Cl]
        M_inv = fisher_selected_inverse  # [E, E]
        V = embeddings_selected  # [N_0, Cl, E]

        for _ in range(len(selected_inds) - n_acquire):
            A_inv = torch.inverse(-I + V @ M_inv @ V.transpose(1, 2))  # [N_i, Cl, Cl]

            # [N_i, Cl, E] @ [E, E] @ [E, E] @ [E, E] @ [N_i, E, Cl] @ [Cl, Cl]
            matrix_prod = (
                V @ M_inv @ fisher_pool @ M_inv @ V.transpose(1, 2) @ A_inv
            )  # [N_i, Cl, Cl]

            trace = torch.vmap(torch.trace)(matrix_prod)  # [N_i,]

            ind_to_remove = torch.argmax(trace).item()  # [1,]
            inds_to_keep = [ind for ind in range(len(V)) if ind != ind_to_remove]  # [N_{i+1},]

            A_inv = torch.inverse(-I + V[ind_to_remove] @ M_inv @ V[ind_to_remove].T)  # [Cl, Cl]

            # [E, E] - [E, E] @ [E, Cl] @ [Cl, Cl] @ [Cl, E] @ [E, E]
            M_inv = M_inv - M_inv @ V[ind_to_remove].T @ A_inv @ V[ind_to_remove] @ M_inv  # [E, E]

            V = V[inds_to_keep]  # [N_{i+1}, Cl, E]

            del selected_inds[ind_to_remove]

        return selected_inds
