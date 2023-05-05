from experiments.user_experience.sa_strategy import CategorySelfAdaptingStrategy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from experiments.user_experience_rl.dispatcher import QNetworkDispatcher


class QNetworkAdaptingStrategy(CategorySelfAdaptingStrategy):
    """TODO:
    """

    def __init__(self, ref_jobs, q_train_interval=1):
        super().__init__(ref_jobs)
        self.q_train_interval = q_train_interval
        self.jobs_since_last_q_training = 0

    def do_adapt(self, ts, dispatcher: 'QNetworkDispatcher', workers, job=None):
        self._advance_ts(ts, dispatcher)
        if job and job.compilation_ok:
            self.add_job_to_duration_predictor(job)

            self.jobs_since_last_q_training += 1
            if self.jobs_since_last_q_training >= self.q_train_interval:
                dispatcher.q_network.train()
                self.jobs_since_last_q_training = 0
        else:
            dispatcher.q_network.update_target_network()
