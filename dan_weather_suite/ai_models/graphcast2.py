import xarray as xr

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout

import numpy as np

import haiku as hk
import functools
import jax


from datetime import timedelta

import dataclasses


fpath1 = "/home/dan/Downloads/dataset_source-hres_date-2022-01-01_res-0.25_levels-13_steps-01.nc"

example_batch = xr.open_dataset(fpath1)

model_path = "/home/dan/Downloads/params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz"

with open(model_path, "rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)

params = ckpt.params
state = {}

model_config = ckpt.model_config
task_config = ckpt.task_config
# target_lead_times = ["6h", "12h", "18h", "24h"]
steps = 1

(
    train_inputs,
    train_targets,
    train_forcings,
) = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", f"{steps*6}h"),
    **dataclasses.asdict(task_config),
)


eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", f"{steps*6}h"),
    **dataclasses.asdict(task_config),
)


print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)


diffs_f = "/home/dan/Downloads/stats_diffs_stddev_by_level.nc"
mean_f = "/home/dan/Downloads/stats_mean_by_level.nc"
stddev_f = "/home/dan/Downloads/stats_stddev_by_level.nc"

diffs_stddev_by_level = xr.load_dataset(diffs_f).compute()
mean_by_level = xr.load_dataset(mean_f).compute()
stddev_by_level = xr.load_dataset(stddev_f).compute()


def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig
):
    """Constructs and wraps the GraphCast Predictor."""
    # Deeper one-step predictor.
    predictor = graphcast.GraphCast(model_config, task_config)

    # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
    # from/to float32 to/from BFloat16.
    predictor = casting.Bfloat16Cast(predictor)

    # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
    # BFloat16 happens after applying normalization to the inputs/targets.
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )

    # Wraps everything so the one-step model can produce trajectories.
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
    return functools.partial(fn, model_config=model_config, task_config=task_config)


# Always pass params and state, so the usage below are simpler
def with_params(fn):
    return functools.partial(fn, params=params, state=state)


# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
    return lambda **kw: fn(**kw)[0]


init_jitted = jax.jit(with_configs(run_forward.init))

run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))


print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings)
