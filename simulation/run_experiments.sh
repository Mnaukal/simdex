#!/usr/bin/bash
for SEED in 1 2 3 4 5 6 7 8 9 10
do
  echo -e "$SEED\n"
  python3 main.py --config ./experiments/user_experience_limits.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz --seed $SEED --output_folder "../results/1S3L/@@config_@@seed"
  python3 main.py --config ./experiments/user_experience_oracle.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz --seed $SEED --output_folder "../results/1S3L/@@config_@@seed"
  python3 main.py --config ./experiments/user_experience_stats.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz --seed $SEED --output_folder "../results/1S3L/@@config_@@seed"
  python3 main.py --config ./experiments/user_experience_zero.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz --seed $SEED --output_folder "../results/1S3L/@@config_@@seed"
  python3 main.py --config ./experiments/user_experience_nn.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz --seed $SEED --output_folder "../results/1S3L/@@config_@@seed"
  python3 main.py --config ./experiments/user_experience_nn_embedding.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz --seed $SEED --output_folder "../results/1S3L/@@config_@@seed"
  python3 main.py --config ./experiments/user_experience_rl_oracle_fast.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz --seed $SEED --output_folder "../results/1S3L/@@config_@@seed"
  python3 main.py --config ./experiments/user_experience_rl_nn_fast.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz --seed $SEED --output_folder "../results/1S3L/@@config_@@seed"
  python3 main.py --config ./experiments/user_experience_rl_embedding_fast.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz --seed $SEED --output_folder "../results/1S3L/@@config_@@seed"
done
echo "fast done"

for SEED in 1 2 3 4 5 6 7 8 9 10
do
  echo -e "$SEED\n"
  python3 main.py --config ./experiments/user_experience_rl_oracle.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz --seed $SEED --output_folder "../results/1S3L/@@config_@@seed"
  python3 main.py --config ./experiments/user_experience_rl_nn.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz --seed $SEED --output_folder "../results/1S3L/@@config_@@seed"
  python3 main.py --config ./experiments/user_experience_rl_embedding.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz --seed $SEED --output_folder "../results/1S3L/@@config_@@seed"
done
