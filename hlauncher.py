from hydra_runners.hydrazine import ExperimentConfig, task_function
from hydra_zen import launch

(jobs,) = launch(
    ExperimentConfig,
    task_function,
    overrides=[
        "hydra/launcher=submitit_slurm",
        "params.rdim_components=2",
        "trainer.gpus=1",
        "trainer.fast_dev_run=True",
        "hydra/job_logging=colorlog", 
        "hydra/hydra_logging=colorlog",
        # "hydra/sweeper=optuna",
        # "hydra.sweeper.direction=maximize",
        # "hydra.sweeper.n_trials=1"
        "hydra.launcher.mem_gb=24",
        "hydra.launcher.tasks_per_node=1",
        "hydra.launcher.nodes=4",
        "hydra.launcher.partition=general",
        "hydra.launcher.timeout_min=30",
        "hydra.launcher.additional_parameters.qos=short" 
    ],
    multirun=True,
)