from hydra_runners.hydrazine import ExperimentConfig, task_function
from hydra_zen import launch

(jobs,) = launch(
    ExperimentConfig,
    task_function,
    overrides=[
        "hydra/launcher=submitit_slurm",
	"params.use_umap=True",
        "params.rdim_components=2, 4, 8, 16, 32",
        "trainer.gpus=1",
        "trainer.fast_dev_run=False",
        "hydra/job_logging=colorlog", 
        "hydra/hydra_logging=colorlog",
        # "hydra/sweeper=optuna",
        # "hydra.sweeper.direction=maximize",
        # "hydra.sweeper.n_trials=4",
        "hydra.launcher.mem_gb=24",
        "hydra.launcher.tasks_per_node=1",
        "hydra.launcher.nodes=1",
        # "hydra.launcher.gpus_per_node=1",
        "+hydra.launcher.additional_parameters.gres=gpu:turing",
        "hydra.launcher.cpus_per_task=6",
        "hydra.launcher.partition=general",
        "hydra.launcher.timeout_min=2700",
        "+hydra.launcher.additional_parameters.qos=short"
    ],
    multirun=True,
)
