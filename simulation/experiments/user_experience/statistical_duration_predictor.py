from interfaces import AbstractDurationPredictor
from jobs import JobDurationIndex


class StatisticalDurationPredictor(AbstractDurationPredictor):

    def __init__(self):
        self.duration_index = JobDurationIndex()

    def predict_duration(self, job):
        # we need to estimate the duration of the job first (! no peeking to job.duration !)
        estimate = self.duration_index.estimate_duration(job.exercise_id, job.runtime_id)
        if estimate is None:
            estimate = job.limits / 2.0
        return estimate

    def add_job(self, job):
        self.duration_index.add(job)
