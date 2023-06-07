from helpers import _create_instance
from workers import WorkerQueue
from interfaces import AbstractDispatcher, AbstractDurationPredictor, AbstractWorkerSelector


class Simulation:
    """Main simulation class. Wraps the algorithm and acts as component container."""

    def __init__(self, configuration, ref_jobs, hash_converters):
        # load parameters from configuration and instantiate necessary components
        self.ref_jobs = ref_jobs[:]
        self.ref_jobs.reverse()

        # metrics
        self.metrics = []
        if "metrics" in configuration:
            for metric in configuration["metrics"]:
                self.metrics.append(_create_instance(metric, ref_jobs, hash_converters))

        # dispatcher
        self.dispatcher: AbstractDispatcher = _create_instance(configuration["dispatcher"], ref_jobs, hash_converters)

        # ML and RL predictors
        if "duration_predictor" in configuration:
            self.duration_predictor: AbstractDurationPredictor = _create_instance(configuration["duration_predictor"], ref_jobs, hash_converters)
        else:
            self.duration_predictor = None

        if "worker_selector" in configuration:
            self.worker_selector: AbstractWorkerSelector = _create_instance(configuration["worker_selector"], ref_jobs, hash_converters)
        else:
            self.worker_selector = None

        # how often System monitoring is called to update the ML models (in seconds)
        self.monitoring_period = float(configuration["period"]) if "period" in configuration else 60.0  # one minute is default

        # simulation state (worker queues)
        if "workers" not in configuration:
            raise RuntimeError("Workers are not specified in the configuration file.")

        self.workers = []
        if isinstance(configuration["workers"], list):
            for worker_attrs in configuration["workers"]:
                self.workers.append(WorkerQueue(**worker_attrs))
        else:
            for i in range(int(configuration["workers"])):
                self.workers.append(WorkerQueue())

        # remaining simulation variables
        self.ts = 0.0  # simulation time
        self.next_monitoring_ts = 0.0  # when the next System monitoring call is scheduled

    def register_metrics(self, *metrics):
        """Additional metrics components may be registered via this method (mainly for debugging purposes)."""
        for m in metrics:
            self.metrics.append(m)

    def __start_simulation(self, ts):
        """Just-in-time initialization."""
        self.ts = ts
        self.next_monitoring_ts = ts + self.monitoring_period

        # initialize injected components
        self.dispatcher.init(self.ts, self.workers)
        if self.duration_predictor:
            self.duration_predictor.init(self.ts)
        if self.worker_selector:
            self.worker_selector.init(self.ts)

        # take an initial snapshot by the metrics collectors
        for metric in self.metrics:
            metric.snapshot(self.ts, self.workers)

    def __advance_time_in_workers(self):
        for worker in self.workers:
            done = worker.advance_time(self.ts)
            for job in done:
                for metric in self.metrics:
                    metric.job_finished(job)
                # invoke System monitoring on ML components
                if self.duration_predictor:
                    self.duration_predictor.system_monitor.job_done(self, job)
                if self.worker_selector:
                    self.worker_selector.system_monitor.job_done(self, job)

    def __advance_time_ref_jobs(self):
        while len(self.ref_jobs) > 0 and self.ref_jobs[-1].spawn_ts + self.ref_jobs[-1].duration <= self.ts:
            job = self.ref_jobs.pop()
            if job.compilation_ok:
                # invoke System monitoring on ML components
                if self.duration_predictor:
                    self.duration_predictor.system_monitor.ref_job_done(self, job)
                if self.worker_selector:
                    self.worker_selector.system_monitor.ref_job_done(self, job)

    def __advance_time(self, ts):
        """Advance the simulation to given point in time, invoking System monitoring periodically."""
        while self.next_monitoring_ts < ts:
            self.ts = self.next_monitoring_ts
            self.__advance_time_in_workers()
            self.__advance_time_ref_jobs()

            # take a measurement for statistics
            for metric in self.metrics:
                metric.snapshot(self.ts, self.workers)

            # invoke System monitoring on ML components
            if self.duration_predictor:
                self.duration_predictor.system_monitor.periodic_monitoring(self)
            if self.worker_selector:
                self.worker_selector.system_monitor.periodic_monitoring(self)
            self.next_monitoring_ts += self.monitoring_period

        self.ts = ts
        self.__advance_time_in_workers()
        self.__advance_time_ref_jobs()

    def run_job(self, job):
        """Advance the simulation up to the point when new job is being spawned and add it to the queues.

        The simulation may perform many internal steps (e.g., invoke periodic system monitoring multiple times) in one run invocation.
        """

        # first run, initialize simulation
        if self.ts == 0.0:
            self.__start_simulation(job.spawn_ts)

        # regular simulation step
        self.__advance_time(job.spawn_ts)
        self.dispatcher.dispatch(job, self.workers, self)
        # invoke System monitoring on ML components
        if self.duration_predictor:
            self.duration_predictor.system_monitor.job_dispatched(self, job)
        if self.worker_selector:
            self.worker_selector.system_monitor.job_dispatched(self, job)

    def end(self):
        # let's wrap up the simulation
        end_ts = self.ts
        for worker in self.workers:
            worker_end_ts = worker.get_finish_ts()
            if worker_end_ts:
                end_ts = max(end_ts, worker.get_finish_ts())
        self.__advance_time(end_ts + self.monitoring_period)
