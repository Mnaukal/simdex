from interfaces import AbstractDurationPredictor
from jobs import JobDurationIndex


class OracleDurationPredictor(AbstractDurationPredictor):

    def predict_duration(self, job):
        # the predictor is cheating here, the duration would not be available until the job is completed !!!
        return job.duration
