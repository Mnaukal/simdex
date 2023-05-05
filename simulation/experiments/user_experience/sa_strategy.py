from interfaces import AbstractSelfAdaptingStrategy, AbstractDispatcherWithDurationPredictor, AbstractDispatcher, \
    AbstractAdaptiveDurationPredictor


class CategorySelfAdaptingStrategy(AbstractSelfAdaptingStrategy):
    """Represents a SA controller that uses simple machine learning.

    Collects job and ref. job metadata to compute categorized statistics of job duration based on their
    affiliation to exercises and runtimes. These statistics are used by dispatcher for predicting the duration
    of incoming jobs.
    """

    def __init__(self, ref_jobs):
        self.ref_jobs = ref_jobs[:]
        self.ref_jobs.reverse()
        self.dispatcher: AbstractDispatcher = ...

    def _advance_ts(self, ts, dispatcher):
        while len(self.ref_jobs) > 0 and self.ref_jobs[-1].spawn_ts + self.ref_jobs[-1].duration <= ts:
            job = self.ref_jobs.pop()
            if job.compilation_ok:
                self.add_job_to_duration_predictor(job)

    def init(self, ts, dispatcher, workers):
        self.dispatcher = dispatcher
        self._advance_ts(ts, dispatcher)

    def do_adapt(self, ts, dispatcher, workers, job=None):
        self._advance_ts(ts, dispatcher)
        if job and job.compilation_ok:
            self.add_job_to_duration_predictor(job)

    def add_job_to_duration_predictor(self, job):
        if isinstance(self.dispatcher, AbstractDispatcherWithDurationPredictor) \
                and isinstance(self.dispatcher.duration_predictor, AbstractAdaptiveDurationPredictor):
            self.dispatcher.duration_predictor.add_job(job)
