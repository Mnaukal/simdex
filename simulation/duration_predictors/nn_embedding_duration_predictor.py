import os
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf

from duration_predictors.nn_duration_predictor import NNDurationPredictor, MLModel, DataProcessor
from constants import RUNTIME_ID_COUNT, EXERCISE_ID_COUNT, TL_GROUP_COUNT
from jobs import ReaderBase


class EmbeddingsMLModel(MLModel):

    def __init__(self, copy_from, **model_params):
        self.embedding_dim = model_params['embedding_dim']
        self.hash_converters = model_params['hash_converters']
        self.embedding_training_data = model_params['embedding_training_data']
        self.embedding_dim = model_params['embedding_dim']
        self.embedding_batch_size = model_params['embedding_batch_size']
        self.embedding_batch_epochs = model_params['embedding_batch_epochs']
        super().__init__(copy_from, **model_params)
        if copy_from is None:
            self._train_embedding()

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
        # self.embedding_model.summary()

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
        print("Training embeddings...")
        self.embedding_model.fit(x, y,
                                 batch_size=self.embedding_batch_size,
                                 epochs=self.embedding_batch_epochs,
                                 verbose=2)
        self.embedding_layer.trainable = False
        print(f"Embeddings training done. {datetime.now()}")


class EmbeddingsDataProcessor(DataProcessor):

    @staticmethod
    def job_to_input(job):
        return [job.exercise_id, job.runtime_id, job.tlgroup_id if hasattr(job, "tlgroup_id") else 0]


class NNEmbeddingDurationPredictor(NNDurationPredictor):
    """Uses neural network regression model to predict the job duration. The model is implemented in TensorFlow."""

    def __init__(self, layer_widths=[64], batch_size=5000, batch_epochs=5, hash_converters=None, embedding_training_data=None, embedding_dim=100, embedding_batch_size=5000, embedding_batch_epochs=20):
        super().__init__(layer_widths, batch_size, batch_epochs)
        self.model_params.update({
            'layer_widths': layer_widths,
            'hash_converters': hash_converters,
            'embedding_training_data': embedding_training_data,
            'embedding_dim': embedding_dim,
            'embedding_batch_size': embedding_batch_size,
            'embedding_batch_epochs': embedding_batch_epochs
        })
        self.data_processor = EmbeddingsDataProcessor()

    def _create_initial_model(self):
        return EmbeddingsMLModel(None, **self.model_params)
