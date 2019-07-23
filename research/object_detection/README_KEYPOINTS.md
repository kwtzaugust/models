# Guide to Keypoint Training with Tensorflow Object Detection API

Training keypoint detection model has been an [open issue](https://github.com/tensorflow/models/issues/4676) since Jul 2018.

The repo uses `legacy/train.py` with data (images, annotations, including keypoint annotations) inputted as TFRs. 

This repo has been modified in 2 key areas to enable keypoint model training:
* `core/post_processing.py` lines 504 to 506 to enable exporting and evaluation of frozen graph  
* `builders/dataset_builder.py` line 131 to enable `tf_example_decoder.py` to enable keypoint data to be parsed from TFR.

The changes have been tested to work with feature extractors but is expected to work for other feature extractors as well:  
* ssd_resnet50_v1_fpn  
* ssd_inception_v2 
* ...to be updated

Example pipelines for the above tested feature extractors have been provided in folder keypoint_examples.
Comments within the `pipeline*.config` files highlight the necessary configurations in the pipeline to enable keypoint training, such as the params `num_keypoints` and `box_code_size`.

An example tfr creator script `txt_to_tfr_retface_v2.py` has been provided that converts WIDERFACE dataset's train images and RetinaFace project's bounding box and keypoint annotations into TFRs that work with `pipeline_wider_kp_dao_incv2.config`.

## How to create TFR with keypoint data
Crucially, tf.Example should be created this way:
```python
# use standard_fields.py for keypoints
example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(im_basename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(source_id.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
        'image/object/keypoint/x': dataset_util.float_list_feature(kx), # essential for keypoints
        'image/object/keypoint/y': dataset_util.float_list_feature(ky), # essential for keypoints
    }))
```
A tf.Example is created for each image, containing the encoded image data, its bounding boxes and keypoints for each bounding box.

In the context of WIDERFACE & RetinaFace, an image may have multiple bounding boxes (for faces) and a set of 5 keypoints (in order: left eye, right eye, nose tip, left mouth tip, right mouth tip) per face.

Suppose an image has 2 face bounding box and 2 sets of 5 facial keypoints. The variables to feed into tf.Example are as follows:
* `xmin`: `[xmin1, xmin2]` where values are float, normalised by image width (or height, for `ymin`, `ymax`)
* Similarly for `xmax`, `ymin`, `ymax`
* `kx`: `[kx11, kx12, kx13, kx14, kx15, kx21, kx22, kx23, kx24, kx25]` where values are float (if keypoint is visible / available), normalised by image width (or height for `ky`) or np.NaN (if keypoint is non-visible / unavailable, as required by core/model.py line 290).

See `txt_to_tfr_retface_v2.py` for an example workflow.

# References
RetinaFace: https://github.com/deepinsight/insightface/tree/master/RetinaFace
WIDERFACE: http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html 

