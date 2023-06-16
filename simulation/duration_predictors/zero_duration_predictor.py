from interfaces import AbstractDurationPredictor, AbstractSystemMonitor


class ZeroDurationPredictor(AbstractDurationPredictor):

    def predict_duration(self, job):
        # we need to estimate the duration of the job first (! no peeking to job.duration !)
        return 0
