from interfaces import AbstractMetricsCollector


class UserExperienceMetricsCollector(AbstractMetricsCollector):
    """User experience metrics attempts to asses user (dis)satisfaction with job delays.

    In general jobs that are expected to be quick (interactive) should not be delayed too long.
    On the other hand, long jobs can be delayed since the used would not wait for them interactively.
    Whether a job is expected to be long or short is also based on ref. solution jobs.
    All jobs are divided into 3 classes -- "ontime" :), "delayed" :|, and "late" :(
    """

    def _add_job_to_index(self, job):
        if job.exercise_id not in self.ref_jobs:
            self.ref_jobs[job.exercise_id] = {"sum": 0.0, "count": 0.0}
        self.ref_jobs[job.exercise_id]["sum"] += job.duration
        self.ref_jobs[job.exercise_id]["count"] += 1.0

        if job.exercise_id not in self.ref_jobs_runtimes:
            self.ref_jobs_runtimes[job.exercise_id] = {}
        if job.runtime_id not in self.ref_jobs_runtimes[job.exercise_id]:
            self.ref_jobs_runtimes[job.exercise_id][job.runtime_id] = {"sum": 0.0, "count": 0.0}
        self.ref_jobs_runtimes[job.exercise_id][job.runtime_id]["sum"] += job.duration
        self.ref_jobs_runtimes[job.exercise_id][job.runtime_id]["count"] += 1.0

    def _get_expected_duration(self, job):
        if job.exercise_id in self.ref_jobs_runtimes and job.runtime_id in self.ref_jobs_runtimes[job.exercise_id]:
            rec = self.ref_jobs_runtimes[job.exercise_id][job.runtime_id]
            return rec["sum"] / rec["count"]

        if job.exercise_id in self.ref_jobs:
            return self.ref_jobs[job.exercise_id]["sum"] / self.ref_jobs[job.exercise_id]["count"]

        if job.compilation_ok:
            return job.duration

        return job.limits

    def __init__(self, ref_jobs, thresholds=[1.0, 2.0]):
        if (ref_jobs is None):
            raise RuntimeError("User experience metrics require ref. jobs to be loaded.")

        self.ref_jobs = {}  # avg duration per exercise_id
        self.ref_jobs_runtimes = {}  # avg duration per exercise_id and runtime_id
        for job in ref_jobs:
            if job.compilation_ok:
                self._add_job_to_index(job)

        # category thresholds as multipliers of expected durations
        self.threshold_ontime, self.threshold_delayed = thresholds

        # collected counters
        self.jobs_ontime = 0
        self.jobs_delayed = 0
        self.jobs_late = 0

    def job_finished(self, job):
        delay = job.start_ts - job.spawn_ts
        expected_duration = self._get_expected_duration(job)
        ontime = max(expected_duration * self.threshold_ontime, 10.0)  # ontime / delayed threshold must be at least 10s
        delayed = max(expected_duration * self.threshold_delayed, 30.0)  # delayed / late threshold must be at least 30s
        if delay <= ontime:
            self.jobs_ontime += 1
        elif delay <= delayed:
            self.jobs_delayed += 1
        else:
            self.jobs_late += 1

        # the metrics is adjusting the expectations of the job duration dynamically
        if job.compilation_ok:
            self._add_job_to_index(job)

    def get_total_jobs(self):
        return self.jobs_ontime + self.jobs_delayed + self.jobs_late

    def print(self):
        print("Total jobs: {}, on time: {}, delayed: {}, late: {}".format(
            self.get_total_jobs(), self.jobs_ontime, self.jobs_delayed, self.jobs_late))
