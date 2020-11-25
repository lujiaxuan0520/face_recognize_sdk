from face_recognize_sdk.face_recognizer import FaceRecognizer

if __name__ == '__main__':
    face_recognizer = FaceRecognizer(training=True)
    face_recognizer.train()
