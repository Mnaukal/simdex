import collections
import numpy as np

from helpers import Timer
from worker_selectors.replay_buffer import ReplayBuffer
from worker_selectors.q_network import DoubleQNetwork
from interfaces import AbstractSystemMonitor, AbstractWorkerSelector


class Inference:

    def __init__(self, epsilon_initial, epsilon_final, epsilon_final_after_jobs):
        self.q_network: DoubleQNetwork = ...
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_final_after_jobs = epsilon_final_after_jobs
        self.dispatched_jobs = 0

    def select_action(self, state):
        q_values = self.q_network.predict_one(state)

        # epsilon-greedy action selection
        epsilon = np.interp(self.dispatched_jobs, [0, self.epsilon_final_after_jobs],
                            [self.epsilon_initial, self.epsilon_final])
        if np.random.uniform() >= epsilon:
            action = np.argmax(q_values)  # greedy
        else:
            action = np.random.randint(len(q_values))

        self.dispatched_jobs += 1
        return action

    def set_model(self, model: DoubleQNetwork):
        self.q_network = model


Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])


class DataStorage:

    def __init__(self, parent: "QNetworkWorkerSelector", replay_buffer_size):
        self.parent = parent
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.incomplete_data = collections.defaultdict(dict)

    def add_state(self, job, state: np.ndarray):
        self.incomplete_data[job]["state"] = state
        self.add_transition_if_complete(job)

    def add_action(self, job, action: int):
        self.incomplete_data[job]["action"] = action
        self.add_transition_if_complete(job)

    def add_reward(self, job, reward: float):
        self.incomplete_data[job]["reward"] = reward
        self.add_transition_if_complete(job)

    def add_next_state(self, job, next_state: np.ndarray):
        self.incomplete_data[job]["next_state"] = next_state
        self.add_transition_if_complete(job)

    def add_transition_if_complete(self, job):
        job_data = self.incomplete_data[job]
        if "state" in job_data and \
                "action" in job_data and \
                "reward" in job_data and \
                "next_state" in job_data:
            self.replay_buffer.append(Transition(job_data["state"], job_data["action"], job_data["reward"], job_data["next_state"]))
            del self.incomplete_data[job]
            self.parent.ml_monitor.transition_added()

    @property
    def transitions_count(self):
        return len(self.replay_buffer)

    def sample_batch(self, batch_size):
        return self.replay_buffer.sample(batch_size)


class DataProcessor:

    class EmptyJob:
        estimated_duration = 0

    @staticmethod
    def get_state(simulation, job):
        workers = simulation.workers
        return np.array([
            job.estimated_duration,
            *[worker.jobs_count() for worker in workers],  # queue lengths (job counts)
            *[np.log2(max(sum(map(lambda j: j.estimated_duration, worker.jobs)), 1)) for worker in workers]  # log of estimated queue lengths (in seconds)
        ])

    def get_next_state(self, simulation, _job):
        # TODO: what is next state? We don't know which job will come next
        # -> use an "empty" job instead (estimated duration = 0)
        empty_job = self.EmptyJob()
        return self.get_state(simulation, empty_job)

    @staticmethod
    def state_size(simulation):
        worker_count = len(simulation.workers)
        return 1 + 2 * worker_count

    @staticmethod
    def compute_reward(_simulation, job):
        waiting = job.start_ts - job.spawn_ts
        if waiting < 10:
            reward = 0
        else:
            reward = -waiting / max(job.duration, 1)
        return reward


class SystemMonitor(AbstractSystemMonitor):

    def __init__(self, parent: 'QNetworkWorkerSelector'):
        self.parent = parent

    def job_done(self, simulation, job):
        reward = self.parent.data_processor.compute_reward(simulation, job)
        self.parent.data_storage.add_reward(job, reward)
        next_state = self.parent.data_processor.get_next_state(simulation, job)
        self.parent.data_storage.add_next_state(job, next_state)

    def periodic_monitoring(self, simulation):
        """The method is called periodically."""
        self.parent.ml_monitor.periodic_update()


class MLMonitor:

    def __init__(self, parent: 'QNetworkWorkerSelector', training_interval, configuration):
        self.parent = parent
        self.training_interval = training_interval
        self.data_since_last_training = 0

        self.inference_timer = Timer("RL Worker Selector inference time", configuration["output_folder"] / "worker_selector_inference_times.csv")
        self.training_timer = Timer("RL Worker Selector training time", configuration["output_folder"] / "worker_selector_training_times.csv")

    def transition_added(self):
        self.data_since_last_training += 1
        if self.data_since_last_training >= self.training_interval:
            self.parent.training.train()
            self.data_since_last_training = 0

    def periodic_update(self):
        self.parent.training.update_target_network()


class Training:

    def __init__(self, parent: 'QNetworkWorkerSelector', batch_size):
        self.parent = parent
        self.batch_size = batch_size

    def train(self):
        if self.parent.data_storage.transitions_count < self.batch_size:
            return

        self.parent.ml_monitor.training_timer.start()
        transitions = self.parent.data_storage.sample_batch(self.batch_size)
        self.parent.q_network.train(transitions)
        self.parent.ml_monitor.training_timer.stop()

    def update_target_network(self):
        self.parent.q_network.update_target_network()


class QNetworkWorkerSelector(AbstractWorkerSelector):

    def __init__(self, epsilon_initial=0.3, epsilon_final=0.01, epsilon_final_after_jobs=10_000, batch_size=64, replay_buffer_size=50_000, training_interval=1, configuration=None, **q_network_args):
        super().__init__(configuration)

        self.q_network_args = q_network_args
        self.q_network: DoubleQNetwork = ...

        self.system_monitor = SystemMonitor(self)
        self.ml_monitor = MLMonitor(self, training_interval, configuration)
        self.data_processor = DataProcessor()
        self.training = Training(self, batch_size)
        self.data_storage = DataStorage(self, replay_buffer_size)
        self.inference = Inference(epsilon_initial, epsilon_final, epsilon_final_after_jobs)

    def init(self, simulation):
        worker_count = len(simulation.workers)
        self.q_network = DoubleQNetwork(inputs_count=DataProcessor.state_size(simulation), actions_count=worker_count, **self.q_network_args)
        self.inference.set_model(self.q_network)

    def end(self, simulation):
        self.ml_monitor.inference_timer.print()
        self.ml_monitor.training_timer.print()
        self.ml_monitor.inference_timer.write()
        self.ml_monitor.training_timer.write()

    def select_worker(self, simulation, job) -> int:
        self.ml_monitor.inference_timer.start()

        state = self.data_processor.get_state(simulation, job)
        worker_index = self.inference.select_action(state)

        self.data_storage.add_state(job, state)
        self.data_storage.add_action(job, worker_index)

        self.ml_monitor.inference_timer.stop()

        return worker_index
