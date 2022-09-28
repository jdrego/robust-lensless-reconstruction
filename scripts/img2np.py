import numpy as np
import glob
import cv2
from PIL import Image
imdir = './data/Synthetic_Dataset/train/'
ext = ['png', 'jpeg','jpg']    # Add image formats here
folder = []
[folder.extend(glob.glob(imdir + 'gt/*.' + e)) for e in ext]
images = [cv2.imread(img, 0) for img in folder]
numOfImgs = len(images)

for ind in range(numOfImgs):
    current_img = folder[ind].split('/')[-1][3:-4]
    print(current_img)
    np.save(imdir + '/gt/np/' + current_img, images[ind])
    print('Saved image:', ind, '/', numOfImgs)