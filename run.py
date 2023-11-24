import argparse
from time import time
from typing import Dict, Type
import numpy as np
import wandb
import yaml


from datasets import dataset_name_to_DatasetClass
from algorithms import algo_name_to_AlgoClass
from metrics import metrics_name_to_MetricsClass



def main():

    # Get the algorithm name and dataset name from the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("algo", help="initialization algorithm to use", choices=algo_name_to_AlgoClass.keys())
    parser.add_argument("dataset", help="dataset to use", choices=dataset_name_to_DatasetClass.keys())
    args = parser.parse_args()
    algo_name = args.algo
    dataset_name = args.dataset
    
    # Get the algorithm class and dataset class from the dictionaries.
    AlgoClass = algo_name_to_AlgoClass[algo_name]
    DatasetClass = dataset_name_to_DatasetClass[dataset_name]
    
    # Load the config, merge it with the command line arguments, and get the relevant values.
    config : dict = yaml.safe_load(open("config.yaml"))
    n_iterations : int = config["n_iterations"]
    do_wandb : bool = config["do_wandb"]
    do_cli : bool = config["do_cli"]
    
    # Create the algorithm, dataset and metric objects using the classes and the config.
    algo = AlgoClass(config["algorithms"][algo_name])
    dataset = DatasetClass(config["datasets"][dataset_name])
    metrics = {metric_name : MetricsClass(config["metrics"][metric_name]) for metric_name, MetricsClass in metrics_name_to_MetricsClass.items()}
    
    # Start the WandB run.
    if do_wandb:
        run_name = f"[{algo_name}]_[{dataset_name}]_[{np.random.randint(1000)}]"
        run = wandb.init(
            project="K-means initialization Benchmark", 
            entity="projet13",
            name=run_name,
            config=config,
            )
    
    
    
    # Get the x dataset
    x_data = dataset.get_x_data()
    
    # Iterate n_iterations times.
    cumulative_training_time = 0
    for iteration in range(n_iterations):
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