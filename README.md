# TODO: title

This is an accompanying repository for a paper called "TODO", which was submitted to the ECSA 2023 Tools & Demos track.

It builds on the "SIMDEX: ReCodEx Backend Simulator and Dataset" artifact ([source code](https://github.com/smartarch/simdex), [paper](https://doi.org/10.1145/3524844.3528078)).

This repository contains:
- A simulator of a job-processing backend of a real system enhanced with machine learning and reinforcement learning components.
- A dataset comprises a log of workloads metadata of real users collected from our instance of [ReCodEx](https://github.com/recodex) (a system for evaluation of coding assignments). The simulator can replay the logs, which provides rather unique evaluation based on real data.


## Getting started

The repository is ready to be used immediately as is (just clone it). You only need to have Python 3.7+ installed. If you are using Python virtual environment, do not forget to adjust paths to `python3` and `pip3` executables.

Install basic dependencies:
```
$> cd ./simulation
$> pip3 install -r ./requirements.txt
```

Quick check the scripts are running (on dataset sample):
```
$> python3 ./main.py --config ./experiments/user_experience_rl_nn_fast.yaml --refs ../data/release01-2021-12-29/ref-solutions.csv ../data/release01-2021-12-29/data-sample.csv
```


## Running prepared experiments

The simulator entry point is `main.py` script which is invoked as:
```
$> python3 ./main.py --config <path-to-config-file> [options] <path-to-data-file>
```
The config is in a `.yaml` file that is used to initialize the simulation. [Config files for our examples](simulation/experiments) are already in this repository and additional information can be found in the [quick guide](simulation).
The data file is `.csv` or `.csv.gz` file that must be in the same format as [our dataset](data).

Additional options recognized by the main script:
- `--refs` option holds one string value -- a path to reference solutions data file (`.csv` or `.csv.gz`), please note that ref. solutions must be loaded for some experiments
- `--limit` option holds one integer, which is a maximal number of rows loaded from the data file (allows to restrict the number of simulated jobs)
- `--progress` is a bool flag that enables progress printouts to std. output (particularly useful for ML experiments that take a long time to process)
- `--seed` option holds one integer which sets the seed for the random number generator
- `--output_folder` specifies the folder where the simulation results will be saved. The output folder path can include special variables which are replaced by their values as follows: `@@config` will be replaced by the name of the config file, `@@datetime` will be replaced by the current date and time, `@@seed` will be replaced with the random generator seed
- `--inference_batch_size` option holds one integer, which defines the batch size for job duration prediction inference (default is `1`). This can be used to speed up the simulation if an NN is used for job duration prediction


## More reading

- [Simulator overview and quick guide to creating your own experiments](simulation)
- [Description of our experiments for the paper](simulation/experiments)
- [Summary of the results](results)
- [Dataset details](data)
