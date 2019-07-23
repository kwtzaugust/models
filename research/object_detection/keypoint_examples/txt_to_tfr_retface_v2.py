'''
About
-----
This script converts WIDERFACE dataset's train images
and RetinaFace's additional keypoint (landmark) annotations
into a TFR that can be used to train keypoint and bounding box (bbox) detectors
with TensorFlow Object Detection API.

Versions
--------
v2
- np.NaN used if keypoint coord is not visible
- Added filter to remove small images.
- If width of a bbox in an img is below thres * width of image, image will be wholly skipped.
- Default thres is calculated like this: 50p / 1920p = 0.026
v2.1
- added description
- cleaned up some code and commentary

Usage
-----
python txt_to_tfr_retface_v2.py \
-opath='WIDER_train_fil0026.record'

Basic Directory Structure
----------------------------
RetinaFace
	- data
		- txt_to_tfr_retface_v2.py
		- retinaface
			- train
				- label.txt
		- WIDERFACE
			- WIDER_train
				- images
					- *--Category
						- *.jpg
		- WIDER_train.record # output tfr file

'''

from __future__ import division, print_function, absolute_import

import os
import io
import sys
import glob
import hashlib
import tensorflow as tf
import time 
import numpy as np

from PIL import Image

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('itxt', 'retinaface/train/label.txt', 'Path to label txt file')
flags.DEFINE_string('idir', 'WIDERFACE/WIDER_train/images', 'Path to dataset folder of jpg')
flags.DEFINE_string('opath', 'WIDER_train.record', 'Path to output TFRecord')
flags.DEFINE_float('thres', 0.026, 'Imgs with at least one bbox_width < thres*im_width will be skipped.')
FLAGS = flags.FLAGS

def toc(tic):
    ''' Weets' progress timer
    Usage:
    Initialise an overall tic = time.time() upstream of main code chunk.
    eg: print("{}: Pipeline prepared.".format(toc(tic))) 
    '''
    return "{:6.2f}s".format(time.time()-tic)

def get_img_dt(txt_file, idir=''):
	# modified version of fn from vis_label_on_img.py
	img_dt = {}

	### Multi-image mode 
	with open(txt_file,'r') as f:
		for line in f:

			line = line.strip()
			### Handle image filepaths
			if line.startswith('#'):
				im_ori_fp = line[1:].strip()
				im_full_fp = '/'.join([idir,im_ori_fp])
				img_dt[im_full_fp] = [] # init empty dict
				assert im_full_fp is not None
				# assert im_ori_fp in anno_dt.keys()
				continue

			### Handle anno lines belonging to curr image
			img_dt[im_full_fp].append(line) # just append the line
	return img_dt

def create_example_from_annoline(im_full_fp, annolines, thres=0.026):
	### Default return variables if unsuccessful 
	example = None 

	### Obtain img data
	im_basename = os.path.basename(im_full_fp)
	# filename = im_basename # has .jpg inside
	source_id = im_full_fp # should be unique

	### Read image
	with tf.gfile.GFile(im_full_fp, 'rb') as fid:
		encoded_jpg = fid.read()
	encoded_jpg_io = io.BytesIO(encoded_jpg)
	image = Image.open(encoded_jpg_io)

	width, height = image.size
	## Check image
	assert width > 0
	assert height > 0
	if image.format != 'JPEG':
		raise ValueError('Image format not JPEG')
	key = hashlib.sha256(encoded_jpg).hexdigest()

	### Prepare containers for bbox + landmarks
	xmin = []
	ymin = []
	xmax = []
	ymax = []
	kx = [] # for keypoints
	ky = []
	kv = []
	
	### Populate bbox and landmark containers
	for line in annolines:
		values = [float(x) for x in line.strip().split()]

		## Get bbox values
		bbox = [values[0], values[1], values[0]+values[2], values[1]+values[3]]
		x1 = bbox[0] # xmin
		y1 = bbox[1] # ymin
		x2 = min(width, bbox[2]) # xmax, but bounded by im width
		y2 = min(height, bbox[3]) # ymax, but bounded by im height
		## Check for impossible situations, skip if present
		if x1>=x2 or y1>=y2:
			print('SKIP: img {} has bbox with impossible dims x1 {} x2 {} y1{} y2{}'.format(
					im_full_fp, x1, x2, y1, y2))
			return None 

		## Filter out small faces below thres
		if (x2-x1) < thres * width:
			print('SKIP: img {} contains bbox of width smaller than thres px {}'.format(im_full_fp, thres*width))
			return None

		xmin.append(float(x1)/width) # normalised
		ymin.append(float(y1)/height)
		xmax.append(float(x2)/width)
		ymax.append(float(y2)/height)

		## Get landmark values after some data prepro
		landmark = np.array( values[4:19], dtype=np.float32 ).reshape((5,3))
		# each landmark[i] is [kx, ky, kv]
		# kv is either -1, 0, or 1 which roughly means toally missing, fully visible, occluded/guessed
		# this ordering doesn't make sense, hence need to swap below
		for li in range(5):
			if landmark[li][0]==-1. and landmark[li][1]==-1.: # missing landmark
				assert landmark[li][2]==-1
				## Convert -1 to NaN as per object_detection.core.model
				# model.py line 290 states: Keypoints are assumed to be provided in normalized coordinates and missing keypoints should be encoded as NaN.
				# IMPT: Missing keypoints should be encoded as NaN
				landmark[li][0]=np.NaN
				landmark[li][1]=np.NaN
			else:
				assert landmark[li][2]>=0
			## Pre-pro: swap landmark vis values
			if landmark[li][2]==0.0: # visible
				landmark[li][2] = 1.0 
			else:
				landmark[li][2] = 0.0

			## Fill container
			# Missing keypoints should be encoded as NaN
			kx.append(float(landmark[li][0])/width) # normalised
			ky.append(float(landmark[li][1])/height)
			kv.append(int(landmark[li][2])) # not essential to TF OD API

	## CUSTOM: Hardcode fill remaining data gaps
	classes = [1 for x in xmin]
	classes_text = ['face'.encode('utf8') for x in xmin]
	truncated = [0 for x in xmin]
	poses = [''.encode('utf8') for x in xmin]
	difficult_obj = [0 for x in xmin]

	### Create TFRecord Example
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
		'image/object/keypoint/x': dataset_util.float_list_feature(kx),
		'image/object/keypoint/y': dataset_util.float_list_feature(ky),
		# 'image/object/keypoint/v': dataset_util.int64_list_feature(kv), # not needed
		
	}))

	return example

def main(_):
	tic = time.time()
	itxt = FLAGS.itxt
	idir = FLAGS.idir

	img_dt = get_img_dt(itxt, idir) # {'im_full_fp1':['bbox_line1', ...], ...}
	
	init = (tf.global_variables_initializer(), tf.local_variables_initializer())
	with tf.Session() as sess:
		sess.run(init)

		with tf.python_io.TFRecordWriter(FLAGS.opath) as writer:
			print('{}: Entering example creation loop. This might take awhile.'.format(toc(tic)))
			prog = 0
			skip_count = 0
			for im_full_fp, annolines in img_dt.items():
				try:
					example = create_example_from_annoline(im_full_fp, annolines, thres=FLAGS.thres)
					if example == None:
						skip_count += 1
						continue # skip img
					writer.write(example.SerializeToString())
					prog += 1
				except (KeyboardInterrupt, SystemExit):sys.exit()
				except:
					print("Unexpected exception occurred on im: {}".format(im_full_fp))
					continue

				if prog % 100 == 0:
					if prog > 100:
						elapsed = time.time() - tic_exp
						print('{}: progress {} @ {:.3f}s/example | skipped {}'.format(toc(tic),prog,elapsed/100,skip_count))
					tic_exp = time.time()

	### Report
	print('{}: produced tfr at {} with {} examples and skipped {}'.format(toc(tic),FLAGS.opath,prog, skip_count))

if __name__ == '__main__':
    tf.app.run()