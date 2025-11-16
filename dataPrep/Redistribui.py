import os
import cv2
import glob

original = "n_person/"
ext = '.jpg'

id = 0

for img_path in glob.glob(original + '/*' + ext):
    img = cv2.imread(img_path)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    if (id < 24000):
        outD = "train/"
    else:
        outD = "test/"

    cv2.imwrite(outD + img_name + ext, img)
    id += 1
