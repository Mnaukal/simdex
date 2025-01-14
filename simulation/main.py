#!/usr/bin/env python3

import sys
import argparse
from datetime import datetime
from pathlib import Path

import ruamel.yaml as yaml

from helpers import init_log, log_with_time, log, close_log, set_random_seed
from interfaces import AbstractBatchedDurationPredictor
from jobs import JobReader, RefJobReader, HashConverter
from simulation import Simulation


def get_configuration(config_file) -> dict:
    with open(config_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print("Simulation config file {} is not in YAML format.".format(config_file))
            print(e)
            exit()


def save_configuration(config_file, configuration):
    with open(config_file, "w") as stream:
        yaml.dump(configuration, stream, default_flow_style=False)


def load_reference_jobs(path, converters):
    reader = RefJobReader(converters=converters)
    reader.open(path)
    jobs = [job for job in reader]
    reader.close()
    return jobs


def create_output_folder(args):
    output_folder = args.output_folder
    if "@@config" in output_folder:
        config_file = Path(args.config)
        output_folder = output_folder.replace("@@config", config_file.stem)
    if "@@seed" in output_folder:
        output_folder = output_folder.replace("@@seed", str(args.seed))
    if "@@datetime" in output_folder:
        output_folder = output_folder.replace("@@datetime", datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to the input .csv or .csv.gz file with jobs log.")
    parser.add_argument("--limit", type=int, default=1000000000,
                        help="Maximal number of jobs to be read from the input file.")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml file with simulation configuration.")
    parser.add_argument("--refs", type=str, required=False,
                        help="Path to .csv or .csv.gz file with log with jobs of reference solutions.")
    parser.add_argument("--progress", default=False, action="store_true",
                        help="If present, progress visualization is on.")
    parser.add_argument("--inference_batch_size", type=int,
                        help="Perform the ML-based prediction for a batch of jobs at the same time to improve performance.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random generator.")
    parser.add_argument("--output_folder", type=str, help="Folder where the output will be saved. '@@config' will be replaced by the name of the config file, '@@datetime' will be replaced by the current date and time, '@@seed' will be replaced with the random generator seed.")
    args = parser.parse_args()

    # initialize the system
    set_random_seed(args.seed)
    configuration = get_configuration(args.config)
    if args.inference_batch_size is None:
        args.inference_batch_size = configuration.get("inference_batch_size", 1)
    if args.output_folder is None:
        args.output_folder = configuration.get("output_folder", "../results/@@config_@@seed")
    configuration["output_folder"] = create_output_folder(args)
    configuration["args"] = args.__dict__
    save_configuration(configuration["output_folder"] / "config.yaml", configuration)

    init_log(configuration["output_folder"])
    configuration["hash_converters"] = {
        "solution_id": HashConverter(),
        "group_id": HashConverter(),
        "tlgroup_id": HashConverter(),
        "exercise_id": HashConverter(),
        "runtime_id": HashConverter(),
    }
    configuration["ref_jobs"] = load_reference_jobs(args.refs, configuration["hash_converters"]) if args.refs else None
    simulation = Simulation(configuration)

    reader = JobReader(converters=configuration["hash_converters"])
    reader.open(args.input_file)

    simulation_start_time = datetime.now()
    log_with_time("Simulation started")

    # read data and run the simulation
    limit = args.limit
    counter = 0
    # use a job buffer for performance optimization
    # this way, we can invoke the neural network for a batch of jobs
    job_buffer = []

    def simulate_jobs(jobs):
        # allow the dispatcher to precompute the predictions
        if args.inference_batch_size > 1 and isinstance(simulation.duration_predictor, AbstractBatchedDurationPredictor):
            simulation.duration_predictor.precompute_batch(jobs)
        # then simulate the jobs sequentially
        for job in jobs:
            simulation.run_job(job)

    # run one job to initialize simulation
    job = next(reader)
    simulation.run_job(job)
    counter += 1

    for job in reader:
        # check the limit
        if limit <= counter:
            break
        counter += 1
        # print progress
        if args.progress and counter % 1_000 == 0:
            sys.stdout.write('.')
            if counter % 50_000 == 0:
                sys.stdout.write('\n')
            sys.stdout.flush()

        # simulate jobs (using the buffer)
        job_buffer.append(job)
        if len(job_buffer) >= args.inference_batch_size:
            simulate_jobs(job_buffer)
            job_buffer = []

    if job_buffer:
        simulate_jobs(job_buffer)

    print()
    simulation.end()
    reader.close()

    simulation_end_time = datetime.now()
    simulation_duration = simulation_end_time - simulation_start_time
    log_with_time("Simulation finished")
    log(f"Simulation duration: {simulation_duration}\n")

    # print out measured statistics
    for metric in simulation.metrics:
        metric.print()

    close_log()

    # save the results for the June 2023 paper
    from results import save_results
    save_results(configuration["output_folder"] / "results.yaml", simulation, simulation_duration)


if __name__ == "__main__":
    main()
