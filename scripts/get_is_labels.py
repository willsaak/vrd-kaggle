import os

from tqdm import tqdm

source_path = '/mnt/renumics-team/ml-competitions/data/datasets/train/challenge-2019-train-vrd.csv'
target_path = '../annotations/is_labels_and_none.csv'
imgs_path = '/mnt/renumics-research/datasets/vis-rel-data/img'

labels_names = ['/m/0cvnqh', '/m/01mzpv', '/m/080hkjn', '/m/0342h', '/m/02jvh9', '/m/04bcr3', '/m/0dt3t', '/m/01940j',
                '/m/071p9', '/m/03ssj5', '/m/04dr76w', '/m/01y9k5', '/m/026t6', '/m/05r5c', '/m/03m3pdh', '/m/01_5g',
                '/m/078n6m', '/m/01s55n', '/m/04ctx', '/m/0584n8', '/m/0cmx8', '/m/02p5f1q', '/m/07y_7']


def get_is_labels():
    lines = []
    with open(source_path) as source, open(target_path, 'w') as target:
        for row in tqdm(source):
            cols = row.split(',')
            if 'is' in cols[-1]:
                # if os.path.isfile(os.path.join(imgs_path, cols[0] + '.jpg')):
                lines.append(cols)
                target.write(row)


def get_none_labels():
    is_labels = []
    with open(target_path) as target:
        for row in tqdm(target):
            line = row.split(',')
            is_labels.append(','.join(line[0:2] + line[3:7]))
    none_labels = []
    with open(source_path) as source:
        for row in tqdm(source):
            cols = row.split(',')
            if os.path.isfile(os.path.join(imgs_path, cols[0] + '.jpg')) and cols[-1] != 'is\n':
                if cols[1] in labels_names:
                    line = ','.join(cols[0:2] + cols[3:7])
                    if line not in is_labels:
                        none_labels.append(','.join(cols[0:2] + ['none'] + cols[3:7] + cols[3:7] + ['is\n']))
                if cols[2] in labels_names:
                    line = ','.join([cols[0]] + [cols[2]] + cols[7:11])
                    if line not in is_labels:
                        none_labels.append(','.join([cols[0]] + [cols[2]] + ['none'] + cols[7:11] + cols[7:11] + ['is\n']))
    with open(target_path, 'a') as target:
        for line in tqdm(none_labels):
            target.write(line)


get_none_labels()
