# Face Recognize SDK

A SDK for face recognizer.

# Get Started
If you just want to judge whether the two images belong to a same person:
+ put the tested images into the directory data/.
+ put the model into the directory checkpoints/.
+ modify the config.py in face_recognize_sdk, mainly including "test_model" and "gpu_devices".
+ take verify_image_pair.py as an example.

If you want to evaluate the model in LFW dataset:
+ download the LFW dataset.
+ put the model into the directory checkpoints/.
+ modify the config.py in face_recognize_sdk, mainly including "test_root", "test_list", "test_model" and "gpu_devices".
+ take evaluate_image_pair.py as an example.

If you want to train a model by yourself or reproduce the results:
+ download the dataset for training.
+ modify the config.py in face_recognize_sdk, mainly including "train_root" and "gpu_devices".
+ take train.py as an example.



