import numpy as np
from interfaces import AbstractMetricsCollector

class JobDelayQuantilesCollector(AbstractMetricsCollector):
    """Metrics collector that computes quantiles of job delays for all jobs."""

    def __init__(self, quantiles=None):
        if quantiles is None:
            self.quantiles = [0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
        self.job_delays = []
        self.jobs = 0

    def job_finished(self, job):
        delay = job.start_ts - job.spawn_ts
        self.job_delays.append(delay)
        self.jobs += 1

    def print(self):
        q_width = max([len(str(q)) for q in self.quantiles])
        print("Delay quantiles:")
        for q, r in zip(self.quantiles, self.compute_quantiles()):
            print(f"    {q:<{q_width}}: {r:.2f}")

    def get_jobs(self):
        return self.jobs

    def compute_quantiles(self):
        return np.quantile(self.job_delays, self.quantiles)
