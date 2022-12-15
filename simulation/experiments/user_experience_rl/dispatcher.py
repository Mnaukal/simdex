import numpy as np

from interfaces import AbstractDispatcher

from experiments.user_experience_rl.sa_strategy import Transition


class JobCategoryDispatcher(AbstractDispatcher):
    def __init__(self, epsilon_initial, epsilon_final, epsilon_final_after_jobs):
        self.q_network = None
        self.replay_buffer = None
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_final_after_jobs = epsilon_final_after_jobs
        self.dispatched_jobs = 0

    def init(self, ts, workers):
        pass

    @staticmethod
    def _get_state(job, workers):
        return [
            job.exercise_id,
            job.runtime_id,
            job.tlgroup_id,
            np.log2(max(job.limits, 1)),
            *[worker.jobs_count() for worker in workers],
            *[np.log2(max(sum(map(lambda j: j.limits, worker.jobs)), 1)) for worker in workers]
        ]

    def dispatch(self, job, workers):
        state = self._get_state(job, workers)
        q_values = self.q_network.predict(np.array([state]))[0]

        epsilon = np.interp(self.dispatched_jobs, [0, self.epsilon_final_after_jobs], [self.epsilon_initial, self.epsilon_final])
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

        self.replay_buffer.append(Transition(state, action, reward, next_state))
