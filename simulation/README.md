# ReCodEx Backend Simulator

This readme presents some internal details of the Simdex, a simulator of [ReCodEx](https://github.com/recodex) backend that replays its [compiled workload logs](../data). For details on how to execute the `main.py` script, please refer to the [main project readme](../README.md#getting-started).

## Scenario

The simulated scenario is rather simple. ReCodEx is a system where students submit their solutions for coding assignments and the backend of the system evaluates them by compiling the submitted code and running a battery of tests on it in a sophisticated sandbox. Each submitted solution becomes a backend job that is assigned to one of the worker servers performing the evaluation. Each worker has a queue in which the jobs are waiting since a worker can only evaluate one job at a time. Jobs are assigned to queues immediately as they are spawned by the job dispatcher and cannot be reassigned later.

## Overview

The simulator algorithm is wrapped in the `Simulation` class in [`simulation.py`](simulation.py). It expects to be called (its public method `run_job`) once for each job by the main loop. Then, the `end` method is called to end the simulation. The simulation class is also responsible for assembling the simulation components from the configuration Yaml file.

Each job is represented by an instance of the `Job` data class (reference solution jobs use the `RefJob` data class). All necessary helper tools for loading data and handling the jobs are in [`jobs.py`](jobs.py). Please note that the readers are built to read jobs one by one (the reader implements an iterator interface), so the whole dataset does not have to be present in memory. On the other hand, this requires that simulated datasets are sorted by the job spawning time.

### Workers and Queues

The backend workers are represented by worker queues in the simulator (i.e., when we are talking about workers or queues, it means the same). Each queue is a simple FIFO that processes the jobs in the exact order in which the jobs are put forth. Job, which is currently at the front of the queue, is being executed (virtually) by the worker. The execution `duration` is taken from the dataset logs. Once the job is finished (stays long enough in the queue), it is removed.

The most important method is
```python
def enqueue(self, job):
```
which places new job into the queue. Please note that the `job` object is altered immediately (its finish timestamp is computed based on the last job in the queue).

Each queue has attributes attached. They are identified by `name` (a string key) and their value may be anything that Python can handle. The attributes are accessed by simple getter and setter:
```python
def get_attribute(self, name):
def set_attribute(self, name, value):
```

Finally, the
```python
def jobs_count(self):
```
will return the number of jobs actually present in the queue (including the front job, which is currently "*running*").

### Dispatchers

The decision to which queue should a job be scheduled is done by a dispatcher.

A dispatcher is classes that must implement `AbstractDispatcher` interface prescribed in [`interfaces.py`](interfaces.py). Namely, it must implement 
```python
def dispatch(self, job, workers, simulation):
```
method which is responsible for job dispatching. It gets a list of all worker queues, selects the right one and places the job inside by calling `worker.enqueue(job)`.

Optionally, it may implement
```python
def init(self, simulation):
```
method that is called once when the dispatcher is being initialized. The `simulation` holds a reference to the `Simulation` object.

#### Predefined dispatchers

The example presented in the paper uses ML and RL for dispatching of the jobs. An ML-based `DurationPredictor` is used to predict the duration of a given job. An RL-based `WorkerSelector` is used to select the worker the job shall be dispatched to.

Currently, there are two dispatchers implemented:

* `DurationFilterDispatcher` ([dispatchers.py](dispatchers.py) module) -- predicts the duration (using the `DurationPredictor` component) of the job and dispatches it to the worker with appropriate duration limit (the maximum allowed job duration) and with the shortest queue
  * requires the `duration_predictor` to be defined in the configuration (see below)
  * requires the `workers` in the configuration to have a `limit` attribute which defines the maximum allowed job duration

* `WorkerSelectorDispatcher` ([dispatchers.py](dispatchers.py) module) -- uses the `WorkerSelector` component to select the worker
  * requires the `worker_selector` to be defined in the configuration (see below)

## Experiments

A more detailed description of the experiments can be found alongside the configurations in the [experiments](experiments) folder.

The results of the experiments are evaluated by metrics.

### Predefined metrics

There are several predefined metric collectors:

* `PowerMetricsCollector` ([default.py](metrics/default.py) module) -- computes total time the queues were active (i.e., power consumption of the workers); it is printed out as relative value (1.0 = one worker queue was active on average). This metric is not used in the experiments for the current paper.

* `JobDelayMetricsCollector` ([default.py](metrics/default.py) module) -- computes average and maximal job delay

* `UserExperienceMetricsCollector` ([user_experience](metrics/user_experience.py) module) -- computes user experience by dividing jobs into three categories:
  - *on time* -- minimal or no delay
  - *delayed* -- noticeable, yet still acceptable delay
  - *late* -- significant (potentially problematic) delay
  
  The output is three numbers -- how many of the jobs felt into each category. The categories are based on duration estimates and multiplication constants (e.g., if job delay is less than `1.5x` its expected duration, it is considered on time). The expected durations are computed from reference solutions jobs (i.e., the `--refs` option must be used when executing the experiment with this metric).

* `UserExperienceMetricsCollectorWithHistory` ([user_experience](metrics/user_experience.py) module) -- extension of the `UserExperienceMetricsCollector` which prints the number of on-time, delayed, and late jobs periodically through the run of the simulation (this helps to see whether there is any trend in the data, e.g. learning takes some time and the results are worse in the beginning of the simulation)

* `JobDelayQuantilesCollector` ([quantile.py](metrics/quantile.py) module) -- computes the quantiles of the job delay

### Defining new metrics

Metric collectors are components that implement an interface defined by `AbstractMetricsCollector`. They may implement two collecting routines.

```python
def snapshot(self, ts, workers):
```
The snapshots are taken periodically, right before the `do_adapt` method of SA strategy is invoked. This method may be used for periodic monitoring of the state of the worker queues (e.g., whether they are active or not).

```python
def job_finished(self, job):
```
This callback is invoked for every job right after it is finished and removed from the worker queue. The collector can use `spawn_ts`, `start_ts`, or even `finish_ts` to gather data regarding the delay of the job.

Finally, the metric component must provide a printing method that outputs the findings.
```python
def print(self, machine=False, verbose=False):
```
The two flags may affect the printing data. The `machine` flag indicates that the output will be collected and processed by a script (probably when a batch of simulations is being executed). The `verbose` flag indicates that the user desires a more detailed output. Both flags may be ignored if not relevant for a particular metric collector.


## Experiment configuration files

Every experiment has to have a configuration `.yaml` file, which holds the initial parameters for the queues and controls the component instantiation process. Examples of the configuration files can be found in the [experiments](experiments) directory.

The configuration file is a collection that holds the following root keys:
- `workers` -- either an integer (number of workers) or a list of collections, each collection is used as an initial set of attributes for one worker
- `dispatcher` -- component specification for the dispatcher
- `duration_predictor` -- component specification for the duration predictor
- `worker_selector` -- component specification for the worker selector
- `period` -- an integer that indicates, how often is the `periodic_monitoring` method is invoked on the `system_monitor` of ML and RL components (in seconds of the simulation time)
- `metrics` -- a list of components specifications of metric modules (all listed modules are used for analysis and their results are printed at the end)

A component specification value is either a string (a full name of the component class), for instance `dispatchers.WorkerSelectorDispatcher` refers to a class `WorkerSelectorDispatcher` in `dispatchers.py` file, or a collection which holds:
- `class` - a full name of the component class as explained above
- `args` - arguments passed to the constructor of that class when it is being instantiated
The `args` can be stored either as a list (positional arguments) or collection (named arguments).

All arguments are treated as static constants; however, in some cases, we need to express the injection pattern as well. For this purpose, we define *injected arguments* as arguments that are replaced with explicit values before being passed to the constructor. The injected arguments are always strings prefixed with `@@`. At the moment, the simulator implements the following injections:
- `@@ref_jobs` -- injects the list of loaded reference solution jobs (requires that `--refs` command line option is used; otherwise the simulation fails)
- `@@hash_converters` -- injects the `HashConverter` instances used to convert the identifiers of jobs, exercises, etc. when loading the data
