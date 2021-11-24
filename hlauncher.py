from hydra_runners.hydrazine import ExperimentConfig, task_function
from hydra_zen import launch

(jobs,) = launch(
    ExperimentConfig,
    task_function,
    overrides=[
        "hydra/launcher=submitit_local",
        "params.batch_size=50, 100, 200",
        "trainer.gpus=0",
        "trainer.fast_dev_run=True",
        "hydra/job_logging=colorlog", 
        "hydra/hydra_logging=colorlog",
        "hydra/sweeper=optuna",
        "hydra.sweeper.direction=maximize",
        "hydra.sweeper.n_trials=3"
    ],
    multirun=True,
)