#!/usr/bin/env python3

import sys
import argparse
from datetime import datetime

import ruamel.yaml as yaml

from interfaces import AbstractDispatcherWithDurationPredictor, AbstractBatchedDurationPredictor
from jobs import JobReader, RefJobReader, HashConverter
from simulation import Simulation


def get_configuration(config_file):
    with open(config_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print("Simulation config file {} is not in YAML format.".format(config_file))
            print(e)
            exit()


def load_reference_jobs(path, converters):
    reader = RefJobReader(converters=converters)
    reader.open(path)
    jobs = [job for job in reader]
    reader.close()
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to the input .csv or .csv.gz file with jobs log.")
    parser.add_argument("--limit", type=int, default=1000000000,
                        help="Maximal number of jobs to be read from the input file.")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml file with simulation configuration.")
    parser.add_argument("--refs", type=str, required=False,
                        help="Path to .csv or .csv.gz file with log with jobs of reference solutions.")
    parser.add_argument("--progress", default=False, action="store_true",
                        help="If present, progress visualization is on.")
    parser.add_argument("--inference_batch_size", type=int, default=500,
                        help="Perform the ML-based prediction for a batch of jobs at the same time to improve performance.")
    args = parser.parse_args()

    # initialize the system
    configuration = get_configuration(args.config)
    hash_converters = {
        "solution_id": HashConverter(),
        "group_id": HashConverter(),
        "tlgroup_id": HashConverter(),
        "exercise_id": HashConverter(),
        "runtime_id": HashConverter(),
    }
    ref_jobs = load_reference_jobs(args.refs, hash_converters) if args.refs else None
    simulation = Simulation(configuration, ref_jobs, hash_converters)

    reader = JobReader(converters=hash_converters)
    reader.open(args.input_file)

    simulation_start_time = datetime.now()
    if args.progress:
        sys.stdout.write(f"Simulation started {simulation_start_time}\n")
        sys.stdout.flush()

    # read data and run the simulation
    limit = args.limit
    counter = 0
    # use a job buffer for performance optimization
    # this way, we can invoke the neural network for a batch of jobs
    job_buffer = []


    def simulate_jobs(jobs):
        # allow the dispatcher to precompute the predictions
        if isinstance(simulation.dispatcher, AbstractDispatcherWithDurationPredictor) and \
                isinstance(simulation.dispatcher.duration_predictor, AbstractBatchedDurationPredictor):
            simulation.dispatcher.duration_predictor.precompute_batch(jobs)
        # then simulate the jobs sequentially
        for job in jobs:
            simulation.run(job)


    # run one job to initialize simulation
    job = next(reader)
    simulation.run(job)
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
    simulation.run(None)  # end the simulation
    reader.close()

    simulation_end_time = datetime.now()
    if args.progress:
        sys.stdout.write(f"Simulation finished: {simulation_end_time}\n")
        sys.stdout.flush()
    sys.stdout.write(f"Simulation duration: {simulation_end_time - simulation_start_time}\n")
    sys.stdout.flush()

    # print out measured statistics
    for metric in simulation.metrics:
        metric.print()
