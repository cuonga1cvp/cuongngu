import os
import shutil
import numpy as np
import cv2
label_train_dir = 'dataset/guns/labels/train'
label_test_dir = 'dataset/guns/labels/test'
image_train_dir = 'dataset/guns/images/train'
image_test_dir = 'dataset/guns/images/test'

label_names = next(os.walk(label_train_dir))[2]
label_paths = [label_train_dir+'/'+aa for aa in label_names]

for i, label_path in enumerate(label_paths):
	image_path = image_train_dir + '/' + label_names[i].split('.')[0] + '.jpeg'
	img = cv2.imread(image_path)
	h,w = img.shape[:2]
	new_content =''
	with open(label_path, 'r') as rlabel:
		content = rlabel.read()
		split_content = content.split('\n')
		nums = split_content[0]
		new_content += '0 '
		for j in range(int(nums)):
			list_coor = split_content[j+1].split(' ')
			list_coor = [int(aa) for aa in list_coor]
			x_min, y_min, x_max, y_max = list_coor
			x_min=int(x_min)
			x_max=int(x_max)
			y_min=int(y_min)
			y_max=int(y_max)
			box_width = x_max - x_min
			box_height = y_max - y_min
			x_center = (x_min + box_width/2)/w
			y_center = (y_min + box_height/2)/h
			box_width = box_width/w
			box_height = box_height/h
			coordinate = str(x_center)+' '+str(y_center)+' '+str(box_width)+' '+str(box_height)
			new_content += coordinate +'\n'

		new_content = new_content[:-1]

	#print(label_path, image_path)
	with open(label_path, 'w') as fw:
		fw.write(new_content)

print('Done convert label train!')


label_names = next(os.walk(label_test_dir))[2]
label_paths = [label_test_dir+'/'+aa for aa in label_names]

for i, label_path in enumerate(label_paths):
	image_path = image_test_dir + '/' + label_names[i].split('.')[0] + '.jpeg'
	img = cv2.imread(image_path)
	h,w = img.shape[:2]
	new_content =''
	with open(label_path, 'r') as rlabel:
		content = rlabel.read()
		split_content = content.split('\n')
		nums = split_content[0]
		new_content += '0 '
		for j in range(int(nums)):
			list_coor = split_content[j+1].split(' ')
			list_coor = [int(aa) for aa in list_coor]
			x_min, y_min, x_max, y_max = list_coor
			x_min=int(x_min)
			x_max=int(x_max)
			y_min=int(y_min)
			y_max=int(y_max)
			box_width = x_max - x_min
			box_height = y_max - y_min
			x_center = (x_min + box_width/2)/w
			y_center = (y_min + box_height/2)/h
			box_width = box_width/w
			box_height = box_height/h
			coordinate = str(x_center)+' '+str(y_center)+' '+str(box_width)+' '+str(box_height)
			new_content += coordinate +'\n'

		new_content = new_content[:-1]

	#print(label_path, image_path)
	with open(label_path, 'w') as fw:
		fw.write(new_content)

print('Done convert label test!')