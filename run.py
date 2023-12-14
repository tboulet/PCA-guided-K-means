import argparse
from time import time
from typing import Dict, Type
import numpy as np
from tqdm import tqdm
import wandb

# Config system
import hydra
from omegaconf import OmegaConf, DictConfig


from datasets import dataset_name_to_DatasetClass
from algorithms import algo_name_to_AlgoClass
from metrics import metrics_name_to_MetricsClass


@hydra.main(config_path="configs", config_name="config_default.yaml")
def main(config : DictConfig):

    # Get the config values from the config object.
    config = OmegaConf.to_container(config, resolve=True)
    algo_name : str = config["algo"]["name"]
    dataset_name : str = config["dataset"]["name"]
    n_iterations : int = config["n_iterations"]
    do_wandb : bool = config["do_wandb"]
    do_cli : bool = config["do_cli"]
    do_tqdm : bool = config["do_tqdm"]
    
    # Get the algorithm class and dataset class from the dictionaries.
    AlgoClass = algo_name_to_AlgoClass[algo_name]
    DatasetClass = dataset_name_to_DatasetClass[dataset_name]

    # Create the algorithm, dataset and metric objects using the classes and the config.
    algo = AlgoClass(config["algo"]["config"])
    dataset = DatasetClass(config["dataset"]["config"])
    metrics = {metric_name : MetricsClass(config["metrics"][metric_name]) for metric_name, MetricsClass in metrics_name_to_MetricsClass.items()}



    # Start the WandB run.
    if do_wandb:
        run_name = f"[{algo_name}]_[{dataset_name}]_[{np.random.randint(1000)}]"
        run = wandb.init(
            project=config["wandb_config"]["project"],
            entity=config["wandb_config"]["entity"],
            name=run_name,
            config=config,
            )

    # Get the x dataset
    x_data = dataset.get_x_data()

    # Iterate n_iterations times.
    cumulative_training_time = 0
    iterator = range(n_iterations) if not do_tqdm else tqdm(range(n_iterations))
    
    for iteration in iterator:
        # Get the clustering result. Measure the time it takes to get the clustering result.
        time_start_training = time()
        clustering_result = algo.fit(x_data=x_data)
        run_time = time() - time_start_training
        cumulative_training_time += run_time

        # Log metrics.
        for metric_name, metric in metrics.items():
            metric_result = metric.compute_metrics(
                dataset=dataset, 
                clustering_result=clustering_result,
                )
            
            if do_wandb:
                cumulative_training_time_in_ms = int(cumulative_training_time * 1000)
                wandb.log(metric_result, step=cumulative_training_time_in_ms)
                wandb.log({"time_training" : run_time, "iteration" : iteration}, step=cumulative_training_time_in_ms)  # additional logging just in case
            if do_cli:
                print(f"Metric results at iteration {iteration} for metric {metric_name}: {metric_result}")

    # Finish the WandB run.
    if do_wandb:
        run.finish()


if __name__ == "__main__":
    main()