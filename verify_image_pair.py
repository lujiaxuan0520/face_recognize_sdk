# -*- coding: utf-8 -*-
from face_recognize_sdk.face_recognizer import FaceRecognizer

sim_threshold = 0.38

if __name__ == "__main__":
    face_recognizer = FaceRecognizer(training=False)

    image_path_1 = 'data/1.png'
    # image_path_1 = 'data/4.jpeg'
    img_1 = face_recognizer.preprocess_single(image_path_1)
    image_path_2 = 'data/2.png'
    # image_path_2 = 'data/6.jpeg'
    img_2 = face_recognizer.preprocess_single(image_path_2)

    # execute verify
    match = face_recognizer.verify(img_1, img_2, sim_threshold=sim_threshold)

    # print
    print('face#`{}` and face#`{}` verification result: {}'.format(image_path_1, image_path_2, match))
