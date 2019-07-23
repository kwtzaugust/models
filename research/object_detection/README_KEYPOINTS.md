# Guide to Keypoint Training with Tensorflow Object Detection API

Training keypoint detection model has been an [open issue](https://github.com/tensorflow/models/issues/4676) since Jul 2018.

The repo uses `legacy/train.py` with data (images, annotations, including keypoint annotations) inputted as TFRs. 

This repo has been modified in 2 key areas to enable keypoint model training:
* `core/post_processing.py` lines 504 to 506 to enable exporting and evaluation of frozen graph  
* `builders/dataset_builder.py` line 131 to enable `tf_example_decoder.py` to parse keypoint data from TFR.

The changes have been tested to work with feature extractors but is expected to work for other feature extractors as well:  
* ssd_resnet50_v1_fpn  
* ssd_inception_v2 
* # to be updated

Example pipelines for the above tested feature extractors have been provided in folder keypoint_examples.
Comments within the `pipeline*.config` files highlight the necessary configurations in the pipeline to enable keypoint training, such as the params `num_keypoints` and `box_code_size`.

An example tfr creator script `txt_to_tfr_retface_v2.py` has been provided that converts WIDERFACE dataset's train images and RetinaFace project's bounding box and keypoint annotations into TFRs that work with `pipeline_wider_kp_dao_incv2.config`.
