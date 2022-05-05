import glob
import os

PATH = 'Log_bbox_ratio_0.5percent_allcoco'
f = open('bbox_0.5_img.txt', 'w')
s = ''
for img in glob.glob(PATH + '/*'):
	s += img.rsplit('/')[-1] + '\n'

f.write(s)
f.close()
