import pandas as pd

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

filtered_df = pd.DataFrame(columns=columns)

running_job_finishes = []

for index, row in jobs_df.iterrows():
    running_job_finishes = list(filter(lambda f: f > row['spawn_ts'], running_job_finishes))
    if len(running_job_finishes) > 0:
        new_row = {}
        for column in columns:
            new_row[column] = row[column]
        filtered_df = filtered_df.append(new_row, ignore_index=True)
    running_job_finishes.append(row['finish_ts'])


filtered_df.to_csv('filtered_data.csv', index=False, sep=';')
