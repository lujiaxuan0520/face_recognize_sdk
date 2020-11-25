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

    # get the test set
    images = unique_image(conf.test_list)
    images = [osp.join(conf.test_root, img) for img in images]
    groups = group_image(images, conf.test_batch_size)  # each of the list is a batch

    # get the features of test set
    feature_dict = dict()
    for group in groups:
        data = face_recognizer.preprocess_batch(group)
        d = face_recognizer.generate_feature_batch(data, group)
        feature_dict.update(d)

    # calculate the FAR, FRR, accuracy and find the best threshold
    FAR_list, FRR_list, FAR, FRR, accuracy, threshold = face_recognizer.compute_criteria(feature_dict, conf.test_list, conf.test_root)

    # draw the FAR, FRR curve
    face_recognizer.plot_far_frr(FAR_list, FRR_list)

    # print the test results
    print(
        "Test Model: {}".format(conf.test_model) + "\n"
        "Accuracy: {}".format(accuracy) + "\n"
        "Threshold: {}".format(threshold) + "\n"
        "FAR: {}".format(FAR) + "\n"
        "FRR: {}".format(FRR) + "\n"
    )
