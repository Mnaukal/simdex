# Results of the experiments

This file contains supplementary material to the Evaluation section of the paper. Please refer to the paper for further details.

## User Experience

Summary of the user experience metrics for approaches with and without online ML and RL. The reported results are means of 10 runs of the simulation (with the standard deviation reported for the avg. delay):

| variant     | on time | delayed | late   | avg. delay (std) |
|-------------|---------|---------|--------|------------------|
| statistics  | 98.75\% | 0.26\%  | 0.99\% | 10.15 s (0.00)   |
| oracle      | 98.81\% | 0.26\%  | 0.93\% | 8.61 s (0.00)    |
| NN          | 98.68\% | 0.24\%  | 1.07\% | 10.17 s (0.25)   |
| RL + oracle | 98.65\% | 0.29\%  | 1.06\% | 16.18 s (4.68)   |
| RL + NN     | 98.59\% | 0.25\%  | 1.17\% | 13.73 s (2.17)   |

## Time Overhead

Summary of the time overhead for approaches with and without online ML and RL. The reported times are means of 10 runs (with the relative standard deviations less than 1.2\% for all the values):

| variant     | dispatch | `JDP` inference | `JDP` training (every 500 jobs) | `WS` inference | `WS` training (every 10 jobs) |
|-------------|----------|-----------------|---------------------------------|----------------|-------------------------------|
| statistics  | 0.01 ms  | --              | --                              | --             | --                            |
| oracle      | 0.01 ms  | --              | --                              | --             | --                            |
| NN          | 2.72 ms  | 2.7 ms          | 560.3 ms                        | --             | --                            |
| RL + oracle | 0.79 ms  | --              | --                              | 0.8 ms         | 140.3 ms                      |
| RL + NN     | 3.72 ms  | 2.9 ms          | 630.9 ms                        | 0.8 ms         | 144.2 ms                      |
