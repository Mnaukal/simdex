import os
from typing import Union, TYPE_CHECKING

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf
import numpy as np

try:
    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
except RuntimeError:  # "Inter op parallelism cannot be modified after initialization."
    pass


if TYPE_CHECKING:
    from q_network_worker_selector import Transition


class DoubleQNetwork:
    def __init__(self, inputs_count, actions_count, layer_widths=[50], gamma=0.99, tau=0.01, learning_rate=0.001):

        self._network = self._construct_model(inputs_count, actions_count, layer_widths, learning_rate)
        self._target_network = self._construct_model(inputs_count, actions_count, layer_widths, learning_rate)

        self.gamma = gamma
        self.tau = tau

        # run the model once with a dummy input to initialize it
        self._network(tf.zeros([1 if s is None else s for s in self._network.input_shape]))
        self._target_network(tf.zeros([1 if s is None else s for s in self._target_network.input_shape]))

    @staticmethod
    def _construct_model(inputs_count, actions_count, layer_widths, learning_rate) -> tf.keras.Model:
        input_layer = tf.keras.layers.Input(inputs_count)

        hidden = input_layer
        for width in layer_widths:
            hidden = tf.keras.layers.Dense(width, activation=tf.nn.relu)(hidden)

        action_values = tf.keras.layers.Dense(actions_count, name="action_values")(hidden)

        model = tf.keras.Model(input_layer, action_values)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.losses.MSE,
        )
        return model

    @tf.function
    def _train_network(self, states: np.ndarray, q_values: np.ndarray) -> None:
        self._network.optimizer.minimize(
            lambda: self._network.compiled_loss(q_values, self._network(states, training=True)),
            var_list=self._network.trainable_variables
        )

    def train(self, transitions: list['Transition']):
        states = np.array([t.state for t in transitions])
        q_values = np.array(self._network.predict(states))
        next_states = np.array([t.next_state for t in transitions])
        q_next = np.array(self._target_network.predict(next_states))

        for i, t in enumerate(transitions):
            q_values[i, t.action] = t.reward + self.gamma * np.max(q_next[i, :])

        self._train_network(states, q_values)

    @tf.function
    def update_target_network(self):
        for target_var, network_var in zip(self._target_network.variables, self._network.variables):
            target_var.assign(target_var * (1 - self.tau) + network_var * self.tau)

    @tf.function
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._network(states)

    def predict_one(self, state: Union[list, np.array]) -> np.ndarray:
        return self.predict(np.array([state]))[0]
