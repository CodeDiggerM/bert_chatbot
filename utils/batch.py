import numpy as np
import torch


def subsequent_mask(size: int) -> torch.BoolTensor:
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype(np.uint8)
    return torch.BoolTensor(mask == 1)


class Batch:
    def __init__(self, source: torch.Tensor, target: torch.Tensor = None, pad: int = 0):
        self.source = source
        self.source_mask = (source == pad)
        if target is not None:
            self.target = target[:, :-1]
            self.target_y = target[:, 1:]
            self.target_mask = self.make_std_mask(self.target, pad)
            self.n_tokens = (self.target == pad).sum()

    @staticmethod
    def make_std_mask(target: torch.Tensor, pad: int) -> torch.Tensor:
        mask = (target == pad).unsqueeze(-2)
        mask = mask & subsequent_mask(target.size(-1)).type_as(mask)
        return mask
