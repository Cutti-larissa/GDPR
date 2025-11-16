import os
import cv2
import json

json_path = "val_person_with_head.json" 
images_dir = "CocoPersonVal"
output_dir_black = "BBox"
output_dir_blur = "Blur"

os.makedirs(output_dir_black, exist_ok=True)
os.makedirs(output_dir_blur, exist_ok=True)

def get_face_box(keypoints):
    pts = []
    
    for i in [0, 1, 2, 3, 4]: # seleciona os 5 primeiros keypoints
        x, y, v = keypoints[i*3:(i+1)*3] # separa eles em cordx cordy e se é vísivel
        if v > 0:
            pts.append((x, y))
    if not pts:
        return None

    xs, ys = zip(*pts)
    
    x_min = min(xs) - 30
    x_max = max(xs) + 30
    y_min = min(ys) - 30
    y_max = max(ys) + 30
    
    return int(x_min), int(y_min), int(x_max), int(y_max)

with open(json_path, "r") as f:
    coco = json.load(f)

id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

ann_by_img = {}
for ann in coco["annotations"]:
    ann_by_img.setdefault(ann["image_id"], []).append(ann)

for img_id, anns in ann_by_img.items():
    file_name = id_to_filename[img_id]
    img_path = os.path.join(images_dir, file_name)
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    img_black = img.copy()
    img_blur = img.copy()

    for ann in anns:
        if "keypoints" not in ann:
            continue

        box = get_face_box(ann["keypoints"])
        if box is None:
            continue

        x1, y1, x2, y2 = box

        cv2.rectangle(img_black, (x1, y1), (x2, y2), (0, 0, 0), -1) 

        roi = img_blur[y1:y2, x1:x2]
        if roi.size > 0:
            blurred = cv2.GaussianBlur(roi, (51, 51), 0) 
            img_blur[y1:y2, x1:x2] = blurred

    cv2.imwrite(os.path.join(output_dir_black, file_name), img_black)
    cv2.imwrite(os.path.join(output_dir_blur, file_name), img_blur)

