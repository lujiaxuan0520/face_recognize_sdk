# -*- coding: utf-8 -*-
import os.path as osp
from face_recognize_sdk.config import Config as conf
from face_recognize_sdk.face_recognizer import FaceRecognizer

sim_threshold = 0.38

def unique_image(pair_list) -> set:
    """Return unique image path in pair_list.txt"""
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    unique = set()
    for pair in pairs:
        id1, id2, _ = pair.split()
        unique.add(id1)
        unique.add(id2)
    return unique


def group_image(images: set, batch) -> list:
    """Group image paths by batch size"""
    images = list(images)
    size = len(images)
    res = []
    for i in range(0, size, batch):
        end = min(batch + i, size)
        res.append(images[i : end])
    return res


if __name__ == "__main__":
    face_recognizer = FaceRecognizer(training=False)
    not_exit_list = []

    # get the test set
    images = unique_image(conf.test_list)
    images = [osp.join(conf.test_root, img) for img in images]
    groups = group_image(images, conf.test_batch_size)  # each of the list is a batch

    # get the features of test set
    feature_dict = dict()
    for group in groups:
        img_paths = group
        for img_path in img_paths:
            if not osp.exists(img_path):
                not_exit_list.append(img_path.split('masked_whn/')[-1])

    new_file = open('masked_whn_pairs_new.txt', 'w')
    with open(conf.test_list, 'r') as fd:
        pairs = fd.readlines()
    for pair in pairs:
        id1, id2, _ = pair.split()
        if id1 not in not_exit_list and id2 not in not_exit_list:
            new_file.write(pair)