import os

from tqdm import tqdm

# source_path = '/mnt/renumics-research/datasets/vis-rel-data/annotations/train-vrd.csv'
source_path = '/mnt/renumics-team/ml-competitions/data/datasets/train/challenge-2019-train-vrd.csv'
target_path = '../annotations/relationship_labels.csv'
# target_path = '../annotations/relationship_labels_and_none.csv'
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

relationships = ('at\n', 'holds\n', 'on\n', 'under\n', 'wears\n',
                 'plays\n', 'hits\n', 'interacts_with\n', 'inside_of\n')


def get_relationship_labels():
    with open(source_path) as source, open(target_path, 'w') as target:
        source.readline()
        for row in tqdm(source):
            cols = row.split(',')
            if cols[-1] in relationships:
                if os.path.isfile(os.path.join(imgs_path, cols[0] + '.jpg')):
                    target.write(row)


def get_none_labels():
    relationship_labels = []
    with open(target_path) as target:
        for row in tqdm(target):
            relationship_labels.append(','.join(row.split(',')[:-1]))

    all_objects = {}
    with open(source_path) as source:
        for row in tqdm(source):
            cols = row.split(',')
            if os.path.isfile(os.path.join(imgs_path, cols[0] + '.jpg')):
                if cols[-1] == 'is\n':
                    image_name = cols[0]
                    if image_name in all_objects.keys():
                        all_objects[image_name].append(','.join([cols[1]] + cols[3:7]))
                    else:
                        all_objects[image_name] = [','.join([cols[1]] + cols[3:7])]
                elif cols[-1] in relationships:
                    image_name = cols[0]
                    if image_name in all_objects.keys():
                        all_objects[image_name].append(','.join([cols[1]] + cols[3:7]))
                        all_objects[image_name].append(','.join([cols[2]] + cols[7:11]))
                    else:
                        all_objects[image_name] = [','.join([cols[1]] + cols[3:7])]
                        all_objects[image_name] = [','.join([cols[2]] + cols[7:11])]

    lines = []
    for image_name, objects in all_objects.items():
        for first in objects:
            first = first.split(',')
            for second in objects:
                print(first, second)
                second = second.split(',')
                if ','.join([first[0], second[0]]) in pairs:
                    row = ','.join([image_name, first[0], second[0], *first[1:], *second[1:]])
                    lines.append(row)
                elif ','.join([second[0], first[0]]) in pairs:
                    row = ','.join([image_name, second[0], first[0], *second[1:], *first[1:]])
                    lines.append(row)

    none_lines = []
    for row in lines:
        if row not in relationship_labels:
            none_lines.append(row + ',none\n')

    with open(target_path, 'a') as target:
        for line in tqdm(none_lines):
            target.write(line)


get_relationship_labels()
# get_none_labels()
