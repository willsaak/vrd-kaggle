import os

from tqdm import tqdm

source_path = '/mnt/renumics-team/ml-competitions/data/datasets/train/challenge-2019-train-vrd.csv'
# source_path = '../annotations/is_labels.csv'
imgs_path = '/mnt/renumics-research/datasets/vis-rel-data/img'

lines = []
with open(source_path) as source:  # , open(target_path, 'w') as target:
    for row in tqdm(source):
        cols = row.split(',')
        if 'is' in cols[-1]:
            # if os.path.isfile(os.path.join(imgs_path, cols[0] + '.jpg')):
            lines.append(cols)
            # target.write(row)

print(len(lines))
print(set([col[1] for col in lines]))
