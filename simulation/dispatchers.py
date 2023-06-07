from interfaces import AbstractDispatcher


class DurationFilterDispatcher(AbstractDispatcher):
    """Dispatcher that tries to improve user experience by placing long jobs in a separate queue (based on estimated duration)."""

    def init(self, ts, workers):
        pass

    @staticmethod
    def duration_filter(est_duration):
        def fnc(w):
            limit = w.get_attribute("limit")
            return limit is None or limit >= est_duration

        return fnc

    def dispatch(self, job, workers, simulation):
        # we need to estimate the duration of the job first (! no peeking to job.duration !)
        estimate = simulation.duration_predictor.predict_duration(job)

        # get all workers marked as active
        active_workers = list(filter(lambda w: w.get_attribute("active"), workers))
        if not active_workers:
            raise RuntimeError("No active workers available, unable to dispatch job.")

        # select workers where the job would fit (estimate duration is under worker limit)
        best_workers = list(filter(self.duration_filter(estimate), active_workers))
        if not best_workers:
            best_workers = active_workers  # fallback, if no worker passes the limit

        best_workers.sort(key=lambda w: w.jobs_count())
        target = best_workers[0]
        target.enqueue(job)


class WorkerSelectorDispatcher(AbstractDispatcher):
    """Dispatcher that uses the simulation.workers_selector to select the best worker."""

    def dispatch(self, job, workers, simulation):
        # we need to estimate the duration of the job first (! no peeking to job.duration !)
        estimate = simulation.duration_predictor.predict_duration(job)

        worker_index = simulation.worker_selector.select_worker(job)  # TODO

        target = workers[worker_index]
        target.enqueue(job)
