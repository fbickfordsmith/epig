"""
B = batch size
E = embedding size
N = number of examples
"""

from typing import Sequence, Tuple

import torch
from laplace import ParametricLaplace
from torch import Tensor
from torch.func import grad, vmap
from torch.utils.data import DataLoader

from src.data.utils import get_next
from src.trainers.pytorch import PyTorchTrainer


class PyTorchClassificationTrainer(PyTorchTrainer):
    def train_step(self, loader: DataLoader) -> dict:
        inputs, labels = get_next(loader)  # [N, ...], [N,]

        self.model.train()

        acc, nll = self.evaluate_train(inputs, labels)  # [1,], [1,]

        self.optimizer.zero_grad()
        nll.backward()
        self.optimizer.step()

        return {"acc": acc.item(), "nll": nll.item()}

    def split_params(self, embedding_params: Sequence[str]) -> Tuple[dict, dict]:
        model = self.model.model if isinstance(self.model, ParametricLaplace) else self.model

        grad_params, no_grad_params = {}, {}

        for name, param in model.named_parameters():
            if name in embedding_params:
                grad_params[name] = param.detach()
            else:
                no_grad_params[name] = param.detach()

        return grad_params, no_grad_params

    def compute_badge_embeddings_v1(
        self, loader: DataLoader, embedding_params: Sequence[str]
    ) -> Tensor:
        self.eval_mode()

        model = self.model.model if isinstance(self.model, ParametricLaplace) else self.model

        embeddings = []

        for inputs, _ in loader:
            pseudolosses = self.compute_badge_pseudoloss_v1(inputs)  # [B,]

            for pseudoloss in pseudolosses:
                # Prevent the grad attribute of each tensor accumulating a sum of gradients.
                model.zero_grad()

                pseudoloss.backward(retain_graph=True)

                embedding_i = []

                for name, param in model.named_parameters():
                    if name in embedding_params:
                        gradient = param.grad.detach().flatten().cpu()  # [E',]
                        embedding_i += [gradient]

                embedding_i = torch.cat(embedding_i)  # [E,]
                embeddings += [embedding_i]  # [E,]

        return torch.stack(embeddings)  # [N, E]

    def compute_badge_embeddings_v2(
        self, loader: DataLoader, embedding_params: Sequence[str]
    ) -> Tensor:
        self.eval_mode()

        grad_params, no_grad_params = self.split_params(embedding_params)

        compute_grad = grad(self.compute_badge_pseudoloss_v2, argnums=1)
        compute_grad = vmap(compute_grad, in_dims=(0, None, None), randomness="same")

        embeddings = []

        for inputs, _ in loader:
            gradient_dict = compute_grad(inputs, grad_params, no_grad_params)

            gradients = []

            for name, gradient in gradient_dict.items():
                gradient = gradient.flatten(start_dim=1).cpu()
                gradients += [gradient]

            gradients = torch.cat(gradients, dim=-1)
            embeddings += [gradients]

        return torch.cat(embeddings)  # [N, E]
