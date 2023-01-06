import collections
import os

from constants import TL_GROUP_COUNT, EXERCISE_ID_COUNT, RUNTIME_ID_COUNT
from experiments.user_experience_rl.replay_buffer import ReplayBuffer
from jobs import ReaderBase

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf
import numpy as np
from interfaces import AbstractSelfAdaptingStrategy


class Network:
    def __init__(self, worker_count, hidden_layers) -> None:

        input_layer = tf.keras.layers.Input(1 + 2 * worker_count)

        # construct the network
        hidden = input_layer
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

    @tf.function
    def update_weights_from(self, other: 'Network', tau) -> None:
        for var, other_var in zip(self._model.variables, other._model.variables):
            var.assign(var * (1 - tau) + other_var * tau)


Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])


class CategorySelfAdaptingStrategy(AbstractSelfAdaptingStrategy):
    """TODO:
    """

    def __init__(self, worker_count, layers_widths_rl=[50], batch_size_rl=64, replay_buffer_size=50_000, gamma=0.99, tau=0.01, layers_widths=[64], batch_size=5000, batch_epochs=5, ref_jobs=None, hash_converters=None, embedding_training_data=None, embedding_dim=100, embedding_batch_size=5000, embedding_batch_epochs=20):
        tf.keras.utils.set_random_seed(42)
        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(8)
        # tf.config.set_visible_devices([], 'GPU')

        # RL
        self.network = Network(worker_count, layers_widths_rl)
        self.network.summary()
        self.target_network = Network(worker_count, layers_widths_rl)
        self.batch_size = batch_size_rl
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.gamma = gamma
        self.tau = tau

        # Duration Embeddings
        self.layers_widths = layers_widths
        self.batch_size = batch_size
        self.batch_epochs = batch_epochs
        self.ref_jobs = ref_jobs[:] if ref_jobs else None
        self.hash_converters = hash_converters
        self.embedding_training_data = embedding_training_data
        self.embedding_dim = embedding_dim
        self.embedding_batch_size = embedding_batch_size
        self.embedding_batch_epochs = embedding_batch_epochs
        self.buffer = []
        self.model = None

    def _train_batch_rl(self):
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

    @staticmethod
    def _get_category_encoding_layer(size):
        return lambda feature: tf.one_hot(feature, size + 1)  # +1 since classes are labeled from 1

    def _construct_embeddings(self):
        exercise_id = tf.keras.Input(shape=(1,), dtype='int32')
        embedding_layer = tf.keras.layers.Embedding(input_dim=EXERCISE_ID_COUNT + 1, input_length=1, output_dim=self.embedding_dim)
        flatten_layer = tf.keras.layers.Flatten()

        def embedding(x):
            return flatten_layer(embedding_layer(x))

        embedded = embedding(exercise_id)

        output_exercise = tf.keras.layers.Dense(EXERCISE_ID_COUNT + 1, activation=tf.nn.softmax, name="exercise_id")(embedded)
        output_tlgroup = tf.keras.layers.Dense(TL_GROUP_COUNT + 1, activation=tf.nn.softmax, name="tlgroup_id")(embedded)

        self.embedding_model = tf.keras.Model(inputs=exercise_id, outputs=[output_exercise, output_tlgroup])
        self.embedding_model.compile(optimizer=tf.optimizers.Adam(), loss=[
            tf.losses.SparseCategoricalCrossentropy(),
            tf.losses.SparseCategoricalCrossentropy()
        ])
        self.embedding_model.summary()

        embedding_layer.trainable = False
        self.embedding_layer = embedding_layer
        return embedding

    def _prepare_inputs(self):
        all_inputs = tf.keras.Input(shape=(3,), dtype='int32')
        embedding = self._construct_embeddings()

        encoded_features = [
            embedding(all_inputs[:, 0])
        ]

        domain_sizes = [RUNTIME_ID_COUNT, TL_GROUP_COUNT]
        for idx in range(0, 2):
            encoding_layer = self._get_category_encoding_layer(domain_sizes[idx])
            encoded_col = encoding_layer(all_inputs[:, idx + 1])
            encoded_features.append(encoded_col)

        return all_inputs, encoded_features

    def _create_model(self, layers_widths):
        all_inputs, encoded_features = self._prepare_inputs()

        last_layer = tf.keras.layers.Concatenate()(encoded_features)
        for width in layers_widths:
            last_layer = tf.keras.layers.Dense(int(width), activation=tf.keras.activations.relu)(last_layer)
        output = tf.keras.layers.Dense(1, tf.keras.activations.exponential)(last_layer)

        model = tf.keras.Model(inputs=all_inputs, outputs=output)
        learning_rate = tf.keras.experimental.CosineDecay(0.01, 10000000)
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=tf.losses.Poisson())
        model.summary()
        return model

    def _jobs_to_tensors(self, jobs):
        x = list(map(self.job_to_input, jobs))
        y = list(map(lambda job: [job.duration], jobs))
        return tf.convert_to_tensor(x, dtype=tf.int32), tf.convert_to_tensor(y, dtype=tf.float32)

    @staticmethod
    def job_to_input(job):
        return [job.exercise_id, job.runtime_id, job.tlgroup_id if hasattr(job, "tlgroup_id") else 0]

    def _advance_ts(self, ts):
        while len(self.ref_jobs) > 0 and self.ref_jobs[-1].spawn_ts + self.ref_jobs[-1].duration <= ts:
            job = self.ref_jobs.pop()
            if job.compilation_ok:
                self.buffer.append(job)


    def _train_batch(self):
        """Take the job buffer and use it as batch for training."""
        if len(self.buffer) > self.batch_size:
            x, y = self._jobs_to_tensors(self.buffer)
            self.model.fit(x, y, batch_size=len(self.buffer), epochs=self.batch_epochs, verbose=False)
            self.buffer = []  # reset the job buffer at the end

    def _train_embedding(self):
        reader = ReaderBase()
        reader.converters = {
            'tlgroup_id': self.hash_converters['tlgroup_id'],
            'exercise_id': self.hash_converters['exercise_id'],
        }
        reader.open(self.embedding_training_data)
        jobs = list(reader)

        self.embedding_layer.trainable = True
        x = tf.convert_to_tensor([job['exercise_id'] for job in jobs], dtype=tf.int32)
        y = (
            tf.convert_to_tensor([job['exercise_id'] for job in jobs], dtype=tf.int32),
            tf.convert_to_tensor([job['tlgroup_id'] for job in jobs], dtype=tf.int32)
        )
        self.embedding_model.fit(x, y,
                                 batch_size=self.embedding_batch_size,
                                 epochs=self.embedding_batch_epochs,
                                 verbose=2)
        self.embedding_layer.trainable = False


    def init(self, ts, dispatcher, workers):
        dispatcher.q_network = self.network
        dispatcher.replay_buffer = self.replay_buffer

        self.model = self._create_model(self.layers_widths)
        self._train_embedding()
        self._advance_ts(ts)
        self._train_batch()

        def predictor(jobs):
            x = np.array([self.job_to_input(job) for job in jobs], dtype='int32')
            return self.model(x, training=False).numpy()

        dispatcher.set_predictor(predictor)

    def do_adapt(self, ts, dispatcher, workers, job=None):
        self._advance_ts(ts)
        if job and job.compilation_ok:
            self.buffer.append(job)
            self._train_batch()

            self._train_batch_rl()
            self.target_network.update_weights_from(self.network, self.tau)
        else:
            # for _ in range(10):
            #     self._train_batch()
            # self.target_network.copy_weights_from(self.network)
            pass
