import pandas as pd

# TODO: for now we only have jobs where more than the worker count are running concurrently,
#  would be better to do some sort of dispatch simulation and get more realistic data

NUM_WORKERS = 4
#LATE_THRESHOLD = 3

jobs_df = pd.read_csv('release01-2021-12-29/data/data.csv', sep=';')

jobs_df["finish_ts"] = jobs_df["spawn_ts"] + jobs_df["duration"]

jobs_df.sort_values(by=['spawn_ts'])

columns = ['submission_id',
           'solution_id',
           'group_id',
           'tlgroup_id',
           'exercise_id',
           'runtime_id',
           'worker_group_id',
           'user_id',
           'spawn_ts',
           'end_ts',
           'limits',
           'cpu_time',
           'correctness',
           'compilation_ok',
           'duration']

clustered_df = pd.DataFrame(columns=columns)

running_jobs = []

for index, row in jobs_df.iterrows():
    running_jobs = list(filter(lambda r_j: r_j['job']['finish_ts'] > row['spawn_ts'], running_jobs))
    running_jobs.append({'job': row, 'in_result': False})

    if len(running_jobs) > NUM_WORKERS:
        for running_job in running_jobs:
            if not running_job['in_result']:
                new_row = {}
                for column in columns:
                    new_row[column] = running_job['job'][column]
                clustered_df = clustered_df.append(new_row, ignore_index=True)
                running_job['in_result'] = True


clustered_df.to_csv('clustered_jobs.csv', index=False, sep=';')
