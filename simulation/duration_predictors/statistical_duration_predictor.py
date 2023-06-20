from interfaces import AbstractDurationPredictor, AbstractSystemMonitor
from jobs import JobDurationIndex


class StatisticalDurationPredictor(AbstractDurationPredictor):

    class SystemMonitor(AbstractSystemMonitor):

        def __init__(self, parent: 'StatisticalDurationPredictor'):
            self.parent = parent

        def job_finished(self, simulation, job):
            self.parent.duration_index.add(job)

        def ref_job_finished(self, simulation, ref_job):
            self.parent.duration_index.add(ref_job)

    def __init__(self, configuration):
        super().__init__(configuration)
        self.duration_index = JobDurationIndex()

    def _init_system_monitor(self):
        return self.SystemMonitor(self)

    def predict_duration(self, job):
        # we need to estimate the duration of the job first (! no peeking to job.duration !)
        estimate = self.duration_index.estimate_duration(job.exercise_id, job.runtime_id)
        if estimate is None:
            estimate = job.limits / 2.0
        return estimate
