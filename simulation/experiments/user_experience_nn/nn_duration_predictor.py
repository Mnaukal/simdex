import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf
import numpy as np

from constants import RUNTIME_ID_COUNT, EXERCISE_ID_COUNT
from interfaces import BatchedDurationPredictor

try:
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
except RuntimeError:  # "Inter op parallelism cannot be modified after initialization."
    pass


class NNDurationPredictor(BatchedDurationPredictor):
    """Uses machine-learning neural-network regression model to predict the job duration.

    The model is trained in SA and used by dispatcher (via estimation function interface).
    The model is implemented in TensorFlow.
    """

    def __init__(self, layers_widths=[64], batch_size=5000, batch_epochs=5, ref_jobs=None):
        super().__init__()

        self.layers_widths = layers_widths
        self.batch_size = batch_size
        self.batch_epochs = batch_epochs
        if ref_jobs:
            self.ref_jobs = ref_jobs[:]
            self.ref_jobs.reverse()
        else:
            self.ref_jobs = None

        self.buffer = []
        self.model: tf.keras.Model = self._create_model()

    def _create_model(self):
        all_inputs, encoded_features = self._prepare_inputs()

        last_layer = tf.keras.layers.Concatenate()(encoded_features)
        for width in self.layers_widths:
            last_layer = tf.keras.layers.Dense(int(width), activation=tf.keras.activations.relu)(last_layer)
        output = tf.keras.layers.Dense(1, tf.keras.activations.exponential)(last_layer)

        model = tf.keras.Model(inputs=all_inputs, outputs=output)
        learning_rate = tf.keras.experimental.CosineDecay(0.01, 10000000)
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=tf.losses.Poisson())
        # model.summary()
        return model

    @staticmethod
    def _prepare_inputs():
        all_inputs = tf.keras.Input(shape=(2,), dtype='int32')
        encoded_features = []
        domain_sizes = [EXERCISE_ID_COUNT, RUNTIME_ID_COUNT]
        for idx in range(0, 2):
            encoding_layer = NNDurationPredictor._get_category_encoding_layer(domain_sizes[idx])
            encoded_col = encoding_layer(all_inputs[:, idx])
            encoded_features.append(encoded_col)

        return all_inputs, encoded_features

    @staticmethod
    def _get_category_encoding_layer(size):
        return lambda feature: tf.one_hot(feature, size + 1)  # +1 since classes are labeled from 1

    def _jobs_to_tensors(self, jobs):
        x = list(map(self.job_to_input, jobs))  # inputs
        y = list(map(lambda job: [job.duration], jobs))  # targets
        return tf.convert_to_tensor(x, dtype=tf.int32), tf.convert_to_tensor(y, dtype=tf.float32)

    @staticmethod
    def job_to_input(job):
        return [job.exercise_id, job.runtime_id]

    def train(self):
        """Take the job buffer and use it as batch for training."""
        if len(self.buffer) > self.batch_size:
            x, y = self._jobs_to_tensors(self.buffer)  # get training data
            self.model.fit(x, y, batch_size=len(self.buffer), epochs=self.batch_epochs, verbose=False)
            self.buffer = []  # reset the job buffer at the end

    def _predict_batch(self, jobs) -> list:
        x = np.array([self.job_to_input(job) for job in jobs], dtype='int32')
        return self.model(x, training=False).numpy()
