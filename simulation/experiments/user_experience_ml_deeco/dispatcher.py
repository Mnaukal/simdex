from ml_deeco.estimators import ValueEstimate, CategoricalFeature, TimeFeature
from ml_deeco.simulation import Component

from constants import EXERCISE_ID_COUNT, RUNTIME_ID_COUNT
from interfaces import AbstractDispatcher


def duration_filter(est_duration):
    def fnc(w):
        limit = w.get_attribute("limit")
        return limit is None or limit >= est_duration
    return fnc


class JobCategoryDispatcher(AbstractDispatcher, Component):

    def jobDurationEstimateBaseline(self, job):
        return job.limits / 2

    jobDurationEstimate = ValueEstimate()\
        .withBaseline(jobDurationEstimateBaseline)\
        .using("jobDurationEstimator")

    @jobDurationEstimate.input(CategoricalFeature(EXERCISE_ID_COUNT + 1))
    def job_exercise_id(self, job):
        return job.exercise_id

    @jobDurationEstimate.input(CategoricalFeature(RUNTIME_ID_COUNT + 1))
    def job_runtime_id(self, job):
        return job.runtime_id

    @jobDurationEstimate.target(TimeFeature())
    def duration(self, job):
        return job.duration  # Note that this function is called only after the job is finished, we can thus use the job.duration value.

    @jobDurationEstimate.inputsValid
    def compilation_ok(self, job):
        return job.compilation_ok

    def init(self, ts, workers):
        pass

    def dispatch(self, job, workers):
        # we need to estimate the duration of the job first (! no peeking to job.duration !)
        estimate = self.jobDurationEstimate(job)

        # get all workers marked as active
        active_workers = list(filter(lambda w: w.get_attribute("active"), workers))
        if not active_workers:
            raise RuntimeError("No active workers available, unable to dispatch job.")

        # select workers where the job would fit (estimate duration is under worker limit)
        best_workers = list(filter(duration_filter(estimate), active_workers))
        if not best_workers:
            best_workers = active_workers  # fallback, if no worker passes the limit

        best_workers.sort(key=lambda w: w.jobs_count())
        target = best_workers[0]
        target.enqueue(job)

        # We also want to collect data for training of the estimate
        JobCategoryDispatcher.jobDurationEstimate.collectInputs(self, job)
        # The following line would be called when the job is finished. As we are in a simulation, the value is already known.
        JobCategoryDispatcher.jobDurationEstimate.collectTargets(self, job)
