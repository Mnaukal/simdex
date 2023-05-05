from interfaces import AbstractAdaptiveDurationPredictor
from jobs import JobDurationIndex


class StatisticalDurationPredictor(AbstractAdaptiveDurationPredictor):

    def __init__(self):
        self.duration_index = JobDurationIndex()

    def predict_duration(self, job):
        # we need to estimate the duration of the job first (! no peeking to job.duration !)
        estimate = self.duration_index.estimate_duration(job.exercise_id, job.runtime_id)
        if estimate is None:
            estimate = job.limits / 2.0
        return estimate

    def add_job(self, job, isRef=False):
        self.duration_index.add(job)

    def train(self):
        pass
