output_path = '/tmp/output.csv'
filenames_path = '../empty'
submission_path = 'submission.csv'

import os
from tqdm import tqdm

filenames = []
outputs = []
with open(output_path) as file:
    for row in tqdm(file):
        outputs.append(row)

with open(filenames_path) as file:
    for row in tqdm(file):
        output = row
        for line in outputs:
            if line.startswith(row.strip()):
                output = line
                break
        filenames.append(output)

with open(submission_path, 'w') as file:
    for line in filenames:
        file.write(line)
