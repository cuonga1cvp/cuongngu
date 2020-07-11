import os
import shutil
import cv2

datasetname = 'jetbot_with_angle'

dir_main = './dataset/'+datasetname
new_dir = dir_main+'/data'

if os.path.exists(new_dir):
	shutil.rmtree(new_dir)
os.mkdir(new_dir)

train_dir = dir_main+'/images/train'
img_names = next(os.walk(train_dir))[2]
img_paths = [train_dir+'/'+aa for aa in img_names]
content = ''
for img_path in img_paths:
	content += img_path + '\n'
content = content[:-1]
with open(new_dir+'/'+datasetname+'_train.txt','w') as fw:
	fw.write(content)

test_dir = dir_main+'/images/test'
img_names = next(os.walk(test_dir))[2]
img_paths = [test_dir+'/'+aa for aa in img_names]
content = ''
for img_path in img_paths:
	content += img_path + '\n'
content = content[:-1]
with open(new_dir+'/'+datasetname+'_test.txt','w') as fw:
	fw.write(content)

with open(new_dir+'/'+datasetname+'.name','w') as fw:
	fw.write(datasetname)

content = 'classes=2' + '\n' + 'train='+new_dir+'/'+datasetname+'_train.txt'+'\n'+'valid='+new_dir+'/'+datasetname+'_test.txt' + '\n' + 'names='+new_dir+'/'+datasetname+'.name'
with open(new_dir+'/'+datasetname+'.data','w') as fw:
	fw.write(content)