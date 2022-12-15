import numpy as np

from interfaces import AbstractDispatcher

from experiments.user_experience_rl.sa_strategy import Transition


class JobCategoryDispatcher(AbstractDispatcher):
    def __init__(self, epsilon_initial, epsilon_final, epsilon_final_after_jobs):
        # RL
        self.q_network = None
        self.replay_buffer = None
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_final_after_jobs = epsilon_final_after_jobs
        self.dispatched_jobs = 0

        # Duration
        self.predictor = None
        self.duration_prediction_cache = {}

    def init(self, ts, workers):
        pass

    def set_predictor(self, predictor):
        self.predictor = predictor

    def predict_duration(self, job):
        if job in self.duration_prediction_cache:
            return float(self.duration_prediction_cache[job])
        else:
            return float(self.predictor([job])[0])

    def precompute_batch(self, jobs):
        predictions = self.predictor(jobs)
        self.duration_prediction_cache = dict(zip(jobs, predictions))

    def _get_state(self, job, workers):
        return [
            self.predict_duration(job),
            *[worker.jobs_count() for worker in workers],
            *[np.log2(max(sum(map(lambda j: j.limits, worker.jobs)), 1)) for worker in workers]
        ]

    def dispatch(self, job, workers):
        state = self._get_state(job, workers)
        q_values = self.q_network.predict(np.array([state]))[0]

        if self.dispatched_jobs % 1000 == 0:
            print(q_values)

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
