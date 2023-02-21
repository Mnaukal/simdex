import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import numpy as np

from interfaces import AbstractDispatcherWithDurationPredictor

from experiments.user_experience_rl.q_network import DoubleQNetwork


class QNetworkDispatcher(AbstractDispatcherWithDurationPredictor):

    def __init__(self, worker_count, epsilon_initial=0.3, epsilon_final=0.01, epsilon_final_after_jobs=10_000, q_network_args={}):
        super().__init__()
        # RL
        self.q_network = DoubleQNetwork(inputs_count=1 + 2 * worker_count, actions_count=worker_count, **q_network_args)
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_final_after_jobs = epsilon_final_after_jobs
        self.dispatched_jobs = 0

    def init(self, ts, workers):
        pass

    def _get_state(self, job, workers):
        return np.array([
            self.duration_predictor.predict_duration(job),
            *[worker.jobs_count() for worker in workers],  # queue lengths (job counts)
            *[np.log2(max(sum(map(lambda j: j.limits, worker.jobs)), 1)) for worker in workers]  # log of queue lengths (in seconds)
        ])

    def dispatch(self, job, workers):
        state = self._get_state(job, workers)
        q_values = self.q_network.predict_one(state)

        if self.dispatched_jobs % 1000 == 0:
            print(q_values)

        epsilon = np.interp(self.dispatched_jobs, [0, self.epsilon_final_after_jobs],
                            [self.epsilon_initial, self.epsilon_final])
        if np.random.uniform() >= epsilon:
            action = np.argmax(q_values)  # greedy
        else:
            action = np.random.randint(len(workers))

        target = workers[action]

        target.enqueue(job)
        self.dispatched_jobs += 1

        # the job.start_ts field is now computed by the simulation
        waiting = job.start_ts - job.spawn_ts
        if waiting < 10:
            reward = 0
        else:
            reward = -waiting / max(job.limits, 1)

        next_state = self._get_state(job, workers)  # TODO: what is next state? We don't know which job will come next

        self.q_network.add_transition(state, action, reward, next_state)
