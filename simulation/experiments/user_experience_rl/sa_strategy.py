from interfaces import AbstractSelfAdaptingStrategy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from experiments.user_experience_rl.dispatcher import QNetworkDispatcher


class QNetworkAdaptingStrategy(AbstractSelfAdaptingStrategy):
    """TODO:
    """

    def __init__(self, ref_jobs):
        self.ref_jobs = ref_jobs[:]
        self.ref_jobs.reverse()

    def _advance_ts(self, ts, dispatcher: 'QNetworkDispatcher'):
        while len(self.ref_jobs) > 0 and self.ref_jobs[-1].spawn_ts + self.ref_jobs[-1].duration <= ts:
            job = self.ref_jobs.pop()
            if job.compilation_ok:
                dispatcher.duration_predictor.add_job(job)

    def init(self, ts, dispatcher, workers):
        self._advance_ts(ts, dispatcher)

    def do_adapt(self, ts, dispatcher, workers, job=None):
        self._advance_ts(ts, dispatcher)
        if job and job.compilation_ok:
            dispatcher.duration_predictor.add_job(job)
            dispatcher.q_network.train()
        else:
            dispatcher.q_network.update_target_network()
