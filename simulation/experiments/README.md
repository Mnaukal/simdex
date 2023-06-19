# Experiments

The example presented in the paper (the **user_experience** scenario) demonstrates application of machine learning (ML) and reinforcement learning (RL). It aims to improve user experience regarding the latency of the jobs. The main assumption is that short jobs should be evaluated interactively whilst long-running jobs may be delayed since the user will not wait for them anyway.

The dispatcher tries to estimate the duration of each job (via a `DurationPredictor`) and place them into appropriate queue. There are several approaches to the duration prediction:
* `limits`-- baseline without any learning; uses half of the job time limits as the duration estimate
* `oracle` -- special baseline that violates simulation causality (uses the actual duration as a prediction); this simulates the ultimate oracle that could precisely predict all the jobs accurately (i.e., the theoretical limit of this scenario)
* `zero`-- baseline which always predicts 0 (zero); serves as a sanity check that the worker configuration makes sense
* `stats` -- simple statistical model; computes an average of durations of past jobs (categorized by exercise ID and runtime ID)
* `nn` -- implements the estimator using simple feedforward neural network (as a regression predictor) implemented using [TensorFlow](https://www.tensorflow.org/)
* `nn_embedding` -- a slightly more sophisticated approach which uses trained embeddings instead of one-hot encoding for the exercise ID (the embeddings are trained as on the exercise ID and the `tlgroup_id`)

Regarding the selection of the worker, two different strategies are utilized:
* `DurationFilterDispatcher` uses one queue dedicated for short jobs only (predicted duration less than `30s`), three more queues open for all jobs. If more than one queue is appropriate for a job, the shortest one (with the least jobs) is used.
* `WorkerSelectorDispatcher` utilizes the `WorkerSelector` component which selects the workers based on RL (specifically, the Deep Q-network algorithm)

More details specific for each configuration can be found in the `.yaml` configuration files in this directory. 
