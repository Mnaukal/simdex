import argparse
import csv
from collections import defaultdict

from jobs import ReaderBase, str_passthru, HashConverter


class Reader(ReaderBase):
    def __init__(self, delimiter=';'):
        super().__init__(delimiter)
        self.converters = {
            "tlgroup_id": HashConverter(),
            "exercise_id": HashConverter(),
            "runtime_id": HashConverter(),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs_file", type=str, help="Path to the input .csv or .csv.gz file with jobs log.")
    parser.add_argument("ref_solutions_file", type=str, help="Path to the input .csv or .csv.gz file with reference solutions.")
    args = parser.parse_args()

    reader = Reader()

    reader.open(args.jobs_file)
    # read all rows
    for row in reader:
        pass
    reader.close()

    print("tl_group unique count:", reader.converters["tlgroup_id"].counter)
    print("exercise_id unique count:", reader.converters["exercise_id"].counter)
    print("runtime_id unique count:", reader.converters["runtime_id"].counter)

    del reader.converters["tlgroup_id"]
    reader.open(args.ref_solutions_file)
    # read all rows
    for row in reader:
        pass
    reader.close()

    print("exercise_id unique count:", reader.converters["exercise_id"].counter)
    print("runtime_id unique count:", reader.converters["runtime_id"].counter)

