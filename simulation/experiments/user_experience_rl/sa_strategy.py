import collections
import os

from experiments.user_experience_rl.replay_buffer import ReplayBuffer

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf
import numpy as np
from interfaces import AbstractSelfAdaptingStrategy


class Network:
    def __init__(self, worker_count, hidden_layers) -> None:

        input_layer = tf.keras.layers.Input(4 + 2 * worker_count)

        # split the inputs
        exercise_id = tf.cast(input_layer[:, 0], tf.int32)
        runtime_id = tf.cast(input_layer[:, 1], tf.int32)
        tlgroup_id = tf.cast(input_layer[:, 2], tf.int32)

        # one-hot encoding
        exercise_id = tf.one_hot(exercise_id, 1875, name="exercise_id")
        runtime_id = tf.one_hot(runtime_id, 20, name="runtime_id")
        tlgroup_id = tf.one_hot(tlgroup_id, 95, name="tlgroup_id")

        categorical_inputs = tf.keras.layers.Concatenate()([exercise_id, runtime_id, tlgroup_id])
        categorical_inputs = tf.keras.layers.Dense(70, activation=tf.nn.relu)(categorical_inputs)

        inputs = tf.keras.layers.Concatenate()([categorical_inputs, input_layer[:, 3:]])

        # construct the network
        hidden = inputs
        for size in hidden_layers:
            hidden = tf.keras.layers.Dense(size, activation=tf.nn.relu)(hidden)

        action_values = tf.keras.layers.Dense(worker_count, name="action_values")(hidden)

        self._model = tf.keras.Model(input_layer, action_values)
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.losses.MSE,
        )

    def summary(self):
        self._model.summary()

    @tf.function
    def train(self, states: np.ndarray, q_values: np.ndarray) -> None:
        self._model.optimizer.minimize(
            lambda: self._model.compiled_loss(q_values, self._model(states, training=True)),
            var_list=self._model.trainable_variables
        )

    @tf.function
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model(states)

    @tf.function
    def copy_weights_from(self, other: 'Network') -> None:
        for var, other_var in zip(self._model.variables, other._model.variables):
            var.assign(other_var)


Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])


class CategorySelfAdaptingStrategy(AbstractSelfAdaptingStrategy):
    """TODO:
    """

    def __init__(self, worker_count, layers_widths=[50], batch_size=64, replay_buffer_size=50_000, gamma=0.99):
        tf.keras.utils.set_random_seed(42)
        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(8)
        # tf.config.set_visible_devices([], 'GPU')

        self.network = Network(worker_count, layers_widths)
        self.network.summary()
        self.target_network = Network(worker_count, layers_widths)
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.gamma = gamma

    def _train_batch(self):
        """Take the job buffer and use it as batch for training."""
        if len(self.replay_buffer) > self.batch_size:
            transitions = self.replay_buffer.sample(self.batch_size, np.random)

            states = np.array([t.state for t in transitions])
            q_values = np.array(self.network.predict(states))
            next_states = np.array([t.next_state for t in transitions])
            q_next = np.array(self.target_network.predict(next_states))

            for i, t in enumerate(transitions):
                q_values[i, t.action] = t.reward + self.gamma * np.max(q_next[i, :])

            self.network.train(states, q_values)

    def init(self, ts, dispatcher, workers):
        dispatcher.q_network = self.network
        dispatcher.replay_buffer = self.replay_buffer

    def do_adapt(self, ts, dispatcher, workers, job=None):
        if job and job.compilation_ok:
            self._train_batch()
        else:
            # for _ in range(10):
            #     self._train_batch()
            self.target_network.copy_weights_from(self.network)
