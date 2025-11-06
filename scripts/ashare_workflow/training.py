"""Model training helpers."""

from __future__ import annotations

from typing import Mapping, Tuple

from qlib.utils import init_instance_by_config
from qlib.workflow import R


def run_training(task_config: Mapping[str, Mapping]) -> Tuple[object, object, object]:
    """Fit the model defined in ``task_config`` and persist the artefacts."""

    model = init_instance_by_config(task_config["model"])
    dataset = init_instance_by_config(task_config["dataset"])
    model.fit(dataset.prepare("train"))
    recorder = R.get_recorder()
    recorder.save_objects(model=model, dataset=dataset)
    return model, dataset, recorder
