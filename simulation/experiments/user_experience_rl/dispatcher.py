import numpy as np

from interfaces import AbstractDispatcher

from experiments.user_experience_rl.sa_strategy import Transition


EPSILON = 0.2


class JobCategoryDispatcher(AbstractDispatcher):
    def __init__(self):
        self.q_network = None
        self.replay_buffer = None

    def init(self, ts, workers):
        pass

    def _get_state(self, job, workers):
        return [
            job.exercise_id,
            job.runtime_id,
            job.tlgroup_id,
            job.limits,
            *[worker.jobs_count() for worker in workers],
            *[sum(map(lambda j: j.limits, worker.jobs)) for worker in workers]
        ]

    def dispatch(self, job, workers):
        state = self._get_state(job, workers)
        q_values = self.q_network.predict(np.array([state]))[0]

        if np.random.uniform() >= EPSILON:
            action = np.argmax(q_values)  # greedy
        else:
            action = np.random.randint(len(workers))

        target = workers[action]

        target.enqueue(job)

        # the job.start_ts field is now computed by the simulation
        waiting = job.start_ts - job.spawn_ts
        reward = -waiting

        next_state = self._get_state(job, workers)  # TODO: what is next state? We don't know which job will come next

        self.replay_buffer.append(Transition(state, action, reward, next_state))
