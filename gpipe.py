import copy
from timeit import timeit
from typing import Iterable, Union, Any

import torch
import torch.nn as nn


def device(d: Union[int, str, torch.device]) -> Union[torch.device]:
    if isinstance(d, torch.device):
        return d
    elif isinstance(d, str):
        return torch.device(d)
    elif isinstance(d, int):
        return torch.device(f'cuda:{d}')


class GPipe(nn.Module):

    def __init__(self, *models: nn.Module, balance: Iterable[int],
                 devices: Union[None, Iterable[Union[int, str, torch.device]]] = None,
                 chunks: Union[None, int] = None) -> None:
        super(GPipe, self).__init__()
        if not devices:
            devices = range(torch.cuda.device_count())
        if chunks is None:
            chunks = len(balance)
        assert chunks > 0
        assert sum(balance) == len(models)

        self.balance = balance
        self.chunks = chunks
        self.devices = [device(d) for d in devices]
        self.models = nn.ModuleList()

        index = 0
        models = list(models)
        for i, bal in enumerate(balance):
            model = models[index:index + bal]
            if bal > 1:
                model = nn.Sequential(*model)
            else:
                model = model[0]
            self.models.append(model.to(self.devices[i]))
            index += bal

    def __len__(self):
        return sum([len(x) for x in self.models])

    def forward(self, x: torch.Tensor) -> Any:
        chunks = [(-i, chunk) for i, chunk in enumerate(x.chunk(self.chunks))]
        chunks = chunks[::-1]
        ret = []
        while len(chunks) > 0:
            new_chunks = []
            for index, chunk in chunks:
                if index >= 0:
                    chunk = self.models[index](chunk.to(self.devices[index]))
                if index == len(self.models) - 1:
                    ret.append(chunk)
                else:
                    new_chunks.append((index + 1, chunk))
            chunks = new_chunks
        return torch.cat(ret)


if __name__ == '__main__':
    data = torch.randn(1000000, 1000)
    linear1 = nn.Linear(1000, 10)
    linear2 = nn.Linear(10, 10)
    linear3 = nn.Linear(10, 10)
    seq = nn.Sequential(linear1, linear2, linear3).cuda(0)
    model2 = GPipe(copy.deepcopy(linear1), copy.deepcopy(linear2), copy.deepcopy(linear3), balance=[1, 2], chunks=2)
    r1 = seq(data.cuda(0))
    r2 = model2(data).cuda(0)
    assert torch.all(torch.eq(r1, r2)).item()
    print(timeit(lambda: seq(data.cuda(0)), number=1))
    print(timeit(lambda: model2(data), number=1))
