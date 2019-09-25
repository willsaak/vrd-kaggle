import os

from tqdm import tqdm

source_path = '/mnt/renumics-team/ml-competitions/data/datasets/train/challenge-2019-train-vrd.csv'
target_path = '../annotations/relationship_labels.csv'
imgs_path = '/mnt/renumics-research/datasets/vis-rel-data/img'

# first_labels = ['/m/0bt9lr', '/m/04dr76w', '/m/04yx4', '/m/0dt3t', '/m/09tvcd',
#                 '/m/02jvh9', '/m/01yrx', '/m/01599', '/m/01_5g', '/m/05r655',
#                 '/m/02p5f1q', '/m/03bt1vf', '/m/01mzpv', '/m/01bl7v', '/m/0cmx8']
# second_labels = ['/m/0hg7b', '/m/0199g', '/m/0bt9lr', '/m/0bwd_0j', '/m/0dt3t',
#                  '/m/09tvcd', '/m/071p9', '/m/01yrx', '/m/04ctx', '/m/04_sv',
#                  '/m/026t6', '/m/01940j', '/m/02p5f1q', '/m/03ssj5', '/m/0dv5r',
#                  '/m/078n6m', '/m/050k8', '/m/05_5p_0', '/m/01599', '/m/0dv9c',
#                  '/m/04dr76w', '/m/01y9k5', '/m/01mzpv', '/m/0h8my_4', '/m/06__v',
#                  '/m/05r5c', '/m/04bcr3', '/m/07y_7', '/m/01s55n', '/m/05ctyq',
#                  '/m/019w40', '/m/03k3r', '/m/03m3pdh', '/m/078jl', '/m/0h2r6',
#                  '/m/0wdt60w', '/m/0l14j_', '/m/080hkjn', '/m/0cvnqh', '/m/01226z',
#                  '/m/0342h', '/m/0k4j', '/m/02jvh9']


pairs = ['/m/04yx4,/m/03k3r', '/m/04yx4,/m/04bcr3', '/m/01bl7v,/m/01y9k5', '/m/05r655,/m/0199g',
         '/m/04yx4,/m/09tvcd', '/m/05r655,/m/0hg7b', '/m/01bl7v,/m/078jl', '/m/05r655,/m/0dt3t',
         '/m/03bt1vf,/m/0342h', '/m/04yx4,/m/0dv9c', '/m/03bt1vf,/m/080hkjn', '/m/03bt1vf,/m/019w40',
         '/m/04dr76w,/m/04bcr3', '/m/03bt1vf,/m/01599', '/m/01_5g,/m/04bcr3', '/m/02p5f1q,/m/01y9k5',
         '/m/05r655,/m/080hkjn', '/m/05r655,/m/026t6', '/m/02jvh9,/m/078n6m', '/m/01599,/m/04bcr3',
         '/m/03bt1vf,/m/03m3pdh', '/m/05r655,/m/01599', '/m/03bt1vf,/m/03k3r', '/m/03bt1vf,/m/0199g',
         '/m/05r655,/m/0bwd_0j', '/m/01mzpv,/m/078n6m', '/m/04yx4,/m/01599', '/m/03bt1vf,/m/0hg7b',
         '/m/03bt1vf,/m/0cvnqh', '/m/01bl7v,/m/02p5f1q', '/m/05r655,/m/0k4j', '/m/0bt9lr,/m/01y9k5',
         '/m/01bl7v,/m/078n6m', '/m/01bl7v,/m/050k8', '/m/03bt1vf,/m/0bt9lr', '/m/04dr76w,/m/01y9k5',
         '/m/01bl7v,/m/01940j', '/m/01yrx,/m/01mzpv', '/m/03bt1vf,/m/04bcr3', '/m/04yx4,/m/0dv5r',
         '/m/05r655,/m/0dv5r', '/m/01bl7v,/m/0k4j', '/m/01bl7v,/m/04dr76w', '/m/01bl7v,/m/01mzpv',
         '/m/01bl7v,/m/026t6', '/m/03bt1vf,/m/01s55n', '/m/04yx4,/m/078jl', '/m/03bt1vf,/m/0dt3t',
         '/m/03bt1vf,/m/07y_7', '/m/04yx4,/m/02jvh9', '/m/03bt1vf,/m/050k8', '/m/04yx4,/m/0wdt60w',
         '/m/01bl7v,/m/04bcr3', '/m/02p5f1q,/m/04bcr3', '/m/01bl7v,/m/03ssj5', '/m/01bl7v,/m/0wdt60w',
         '/m/02p5f1q,/m/078n6m', '/m/05r655,/m/04dr76w', '/m/04yx4,/m/04_sv', '/m/01bl7v,/m/05r5c',
         '/m/05r655,/m/04_sv', '/m/04yx4,/m/0h8my_4', '/m/01mzpv,/m/04bcr3', '/m/03bt1vf,/m/01mzpv',
         '/m/01yrx,/m/01y9k5', '/m/01bl7v,/m/0bt9lr', '/m/03bt1vf,/m/01940j', '/m/05r655,/m/01yrx',
         '/m/01bl7v,/m/01599', '/m/04yx4,/m/01yrx', '/m/03bt1vf,/m/026t6', '/m/04yx4,/m/050k8',
         '/m/04yx4,/m/071p9', '/m/04yx4,/m/0bwd_0j', '/m/05r655,/m/02jvh9', '/m/05r655,/m/050k8',
         '/m/05r655,/m/078n6m', '/m/04yx4,/m/0bt9lr', '/m/01bl7v,/m/0199g', '/m/05r655,/m/0l14j_',
         '/m/0bt9lr,/m/03ssj5', '/m/04dr76w,/m/078n6m', '/m/04yx4,/m/0k4j', '/m/01yrx,/m/03ssj5',
         '/m/04yx4,/m/01s55n', '/m/0dt3t,/m/04bcr3', '/m/01bl7v,/m/0342h', '/m/0bt9lr,/m/03m3pdh',
         '/m/03bt1vf,/m/03ssj5', '/m/04yx4,/m/01y9k5', '/m/05r655,/m/05r5c', '/m/01bl7v,/m/071p9',
         '/m/05r655,/m/03k3r', '/m/01bl7v,/m/0cvnqh', '/m/03bt1vf,/m/0bwd_0j', '/m/03bt1vf,/m/04_sv',
         '/m/05r655,/m/03m3pdh', '/m/0bt9lr,/m/04bcr3', '/m/04yx4,/m/026t6', '/m/03bt1vf,/m/0l14j_',
         '/m/04yx4,/m/02p5f1q', '/m/09tvcd,/m/01y9k5', '/m/04yx4,/m/05ctyq', '/m/05r655,/m/0342h',
         '/m/04yx4,/m/05r5c', '/m/04yx4,/m/0342h', '/m/04yx4,/m/080hkjn', '/m/0cmx8,/m/04bcr3',
         '/m/03bt1vf,/m/0dv5r', '/m/01bl7v,/m/0dv5r', '/m/04yx4,/m/03m3pdh', '/m/01bl7v,/m/01226z',
         '/m/05r655,/m/0cvnqh', '/m/04yx4,/m/06__v', '/m/05r655,/m/05_5p_0', '/m/05r655,/m/01940j',
         '/m/0bt9lr,/m/01mzpv', '/m/05r655,/m/07y_7', '/m/04yx4,/m/019w40', '/m/03bt1vf,/m/0h8my_4',
         '/m/03bt1vf,/m/04dr76w', '/m/04yx4,/m/0199g', '/m/02jvh9,/m/04bcr3', '/m/04yx4,/m/04dr76w',
         '/m/04yx4,/m/01mzpv', '/m/03bt1vf,/m/06__v', '/m/05r655,/m/09tvcd', '/m/01bl7v,/m/01yrx',
         '/m/04yx4,/m/01226z', '/m/04yx4,/m/03ssj5', '/m/03bt1vf,/m/01yrx', '/m/01yrx,/m/04bcr3',
         '/m/05r655,/m/04bcr3', '/m/01bl7v,/m/02jvh9', '/m/03bt1vf,/m/05_5p_0', '/m/03bt1vf,/m/01y9k5',
         '/m/01599,/m/078n6m', '/m/05r655,/m/019w40', '/m/04yx4,/m/04ctx', '/m/05r655,/m/0bt9lr',
         '/m/01bl7v,/m/019w40', '/m/05r655,/m/03ssj5', '/m/03bt1vf,/m/01226z', '/m/05r655,/m/01mzpv',
         '/m/01mzpv,/m/01y9k5', '/m/09tvcd,/m/078n6m', '/m/01bl7v,/m/07y_7', '/m/05r655,/m/01y9k5',
         '/m/01yrx,/m/04_sv', '/m/03bt1vf,/m/09tvcd', '/m/04yx4,/m/078n6m', '/m/0bt9lr,/m/0k4j',
         '/m/09tvcd,/m/04bcr3', '/m/04yx4,/m/0cvnqh', '/m/03bt1vf,/m/0k4j', '/m/03bt1vf,/m/078n6m',
         '/m/04yx4,/m/07y_7', '/m/04yx4,/m/01940j', '/m/0bt9lr,/m/0cvnqh', '/m/04yx4,/m/0hg7b',
         '/m/04yx4,/m/05_5p_0', '/m/04yx4,/m/0h2r6', '/m/03bt1vf,/m/02jvh9', '/m/03bt1vf,/m/02p5f1q']
pairs = []
with open(source_path) as source:
    for row in tqdm(source):
        cols = row.split(',')
        if os.path.isfile(os.path.join(imgs_path, cols[0] + '.jpg')):
            if cols[-1] != 'is\n':
                pairs.append(','.join(cols[1:3]))
print()

# labels_names = ['/m/0cvnqh', '/m/01mzpv', '/m/080hkjn', '/m/0342h', '/m/02jvh9', '/m/04bcr3', '/m/0dt3t', '/m/01940j',
#                 '/m/071p9', '/m/03ssj5', '/m/04dr76w', '/m/01y9k5', '/m/026t6', '/m/05r5c', '/m/03m3pdh', '/m/01_5g',
#                 '/m/078n6m', '/m/01s55n', '/m/04ctx', '/m/0584n8', '/m/0cmx8', '/m/02p5f1q', '/m/07y_7']
#
#
# def get_is_labels():
#     lines = []
#     with open(source_path) as source, open(target_path, 'w') as target:
#         for row in tqdm(source):
#             cols = row.split(',')
#             if 'is' in cols[-1]:
#                 # if os.path.isfile(os.path.join(imgs_path, cols[0] + '.jpg')):
#                 lines.append(cols)
#                 target.write(row)
#
#
# def get_none_labels():
#     is_labels = []
#     with open(target_path) as target:
#         for row in tqdm(target):
#             line = row.split(',')
#             is_labels.append(','.join(line[0:2] + line[3:7]))
#     none_labels = []
#     with open(source_path) as source:
#         for row in tqdm(source):
#             cols = row.split(',')
#             if os.path.isfile(os.path.join(imgs_path, cols[0] + '.jpg')) and cols[-1] != 'is\n':
#                 if cols[1] in labels_names:
#                     line = ','.join(cols[0:2] + cols[3:7])
#                     if line not in is_labels:
#                         none_labels.append(','.join(cols[0:2] + ['none'] + cols[3:7] + cols[3:7] + ['is\n']))
#                 if cols[2] in labels_names:
#                     line = ','.join([cols[0]] + [cols[2]] + cols[7:11])
#                     if line not in is_labels:
#                         none_labels.append(','.join([cols[0]] + [cols[2]] + ['none'] + cols[7:11] + cols[7:11] + ['is\n']))
#     with open(target_path, 'a') as target:
#         for line in tqdm(none_labels):
#             target.write(line)
#
#
# get_none_labels()
