from interfaces import AbstractDispatcherWithDurationPredictor


def duration_filter(est_duration):
    def fnc(w):
        limit = w.get_attribute("limit")
        return limit is None or limit >= est_duration
    return fnc


class DurationFilterDispatcher(AbstractDispatcherWithDurationPredictor):
    """Dispatcher that tries to improve user experience by placing long jobs in a separate queue.

    The estimation whether a job will be short or long is based on jobs that were already evaluated
    (both regular and ref.). The SA strategy is responsible for filling data for the estimator.
    """

    def init(self, ts, workers):
        pass

    def dispatch(self, job, workers):
        # we need to estimate the duration of the job first (! no peeking to job.duration !)
        estimate = self.duration_predictor.predict_duration(job)

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
