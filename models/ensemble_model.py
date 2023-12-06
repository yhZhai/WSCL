from typing import Dict, List

import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    def __init__(self, models: Dict, mvc_single_weight: Dict):
        super().__init__()

        self.sub_models = nn.ModuleDict(models)
        self.modality = list(self.sub_models.keys())
        self.mvc_single_weight = mvc_single_weight
        for k, v in self.mvc_single_weight.items():
            assert 0 <= v <= 1, "The weight of {} for {} is out of range".format(v, k)

    def forward(self, image, seg_size=None):
        result = {}
        for modality in self.modality:
            result[modality] = self.sub_models[modality](image, seg_size)

        avg_result = {}
        for k in result[self.modality[0]].keys():
            avg_result[k] = torch.zeros_like(result[self.modality[0]][k])
            for modality in self.modality:
                avg_result[k] = (
                    avg_result[k]
                    + self.mvc_single_weight[modality] * result[modality][k]
                )
        result["ensemble"] = avg_result

        return result
