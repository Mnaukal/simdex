import argparse
import csv
from collections import defaultdict

from jobs import ReaderBase, str_passthru


class Reader(ReaderBase):
    def __init__(self, delimiter=';'):
        super().__init__(delimiter)
        self.converters = {
            "tlgroup_id": str_passthru,
            "exercise_id": str_passthru,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to the input .csv or .csv.gz file with jobs log.")
    parser.add_argument("output_file", type=str, help="Path to the output .csv.")
    args = parser.parse_args()

    reader = Reader()
    reader.open(args.input_file)

    tlgroups = defaultdict(set)

    for row in reader:
        tlgroup_id = row["tlgroup_id"]
        exercise_id = row["exercise_id"]
        tlgroups[tlgroup_id].add(exercise_id)

    reader.close()

    writer = csv.writer(open(args.output_file, "w", newline=''), delimiter=';')
    writer.writerow(["tlgroup_id", "exercise_id"])
    for tlgroup_id in tlgroups:
        for exercise_id in tlgroups[tlgroup_id]:
            writer.writerow([tlgroup_id, exercise_id])
