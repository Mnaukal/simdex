import os

from jobs import ReaderBase

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf
import numpy as np
from interfaces import AbstractSelfAdaptingStrategy


class CategorySelfAdaptingStrategy(AbstractSelfAdaptingStrategy):
    """Uses machine-learning neural-network regression model to predict the job duration.

    The model is trained in SA and used by dispatcher (via estimation function interface).
    The model is implemented in TensorFlow.
    """

    def __init__(self, layers_widths=[64], batch_size=5000, batch_epochs=5, ref_jobs=None, hash_converters=None, embedding_training_data=None, embedding_dim=100, embedding_batch_size=5000, embedding_batch_epochs=20):
        tf.config.threading.set_inter_op_parallelism_threads(4)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        # tf.config.set_visible_devices([], 'GPU')

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

    @staticmethod
    def _get_category_encoding_layer(size):
        return lambda feature: tf.one_hot(feature, size + 1)  # +1 since classes are labeled from 1

    def _construct_embeddings(self):
        exercise_id = tf.keras.Input(shape=(1,), dtype='int32')
        embedding_layer = tf.keras.layers.Embedding(input_dim=1876, input_length=1, output_dim=self.embedding_dim)
        flatten_layer = tf.keras.layers.Flatten()

        def embedding(x):
            return flatten_layer(embedding_layer(x))

        embedded = embedding(exercise_id)

        output_exercise = tf.keras.layers.Dense(1876, activation=tf.nn.softmax, name="exercise_id")(embedded)
        output_tlgroup = tf.keras.layers.Dense(96, activation=tf.nn.softmax, name="tlgroup_id")(embedded)

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

        domain_sizes = [20, 95]  # runtime_id, tlgroup_id
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
        self.model = self._create_model(self.layers_widths)
        self._train_embedding()
        self._advance_ts(ts)
        self._train_batch()

        # @tf.function
        # def predict_single(input):
        #     return self.model(input, training=False)[0]
        #
        # def predictor(job):
        #     x = np.array([[job.exercise_id, job.runtime_id]], dtype='int32')
        #     return predict_single(x).numpy()[0]

        def predictor(jobs):
            x = np.array([self.job_to_input(job) for job in jobs], dtype='int32')
            return self.model(x, training=False).numpy()

        dispatcher.set_predictor(predictor)

    def do_adapt(self, ts, dispatcher, workers, job=None):
        self._advance_ts(ts)
        if job and job.compilation_ok:
            self.buffer.append(job)
            self._train_batch()
