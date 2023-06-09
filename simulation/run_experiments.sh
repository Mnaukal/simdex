#!/usr/bin/bash
python3 main.py --config ./experiments/user_experience_limits.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz
python3 main.py --config ./experiments/user_experience_oracle.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz
python3 main.py --config ./experiments/user_experience_stats.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz
python3 main.py --config ./experiments/user_experience_nn.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz
python3 main.py --config ./experiments/user_experience_nn_embedding.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz
python3 main.py --config ./experiments/user_experience_rl_oracle_fast.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz
python3 main.py --config ./experiments/user_experience_rl_embedding_fast.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz
python3 main.py --config ./experiments/user_experience_rl_oracle.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz
python3 main.py --config ./experiments/user_experience_rl_embedding.yaml --refs ../data/release02-2023-01-04/ref-solutions.csv ../data/release02-2023-01-04/data.csv.gz
