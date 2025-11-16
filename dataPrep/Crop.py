import os
import json
from PIL import Image

json_path = "val_person_with_head.json"
images_dir = "person_with_head_val"
output_dir = "PersonValCrop"

os.makedirs(output_dir, exist_ok=True)

with open(json_path, "r") as f:
    coco = json.load(f)

id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

id_to_category = {cat["id"]: cat["name"] for cat in coco["categories"]}

for ann in coco["annotations"]:
    image_id = ann["image_id"]
    bbox = ann["bbox"]  # [x, y, w, h]
    category_id = ann["category_id"]
    
    file_name = id_to_filename[image_id]
    img_path = os.path.join(images_dir, file_name)
    
    if not os.path.exists(img_path):
        print(f"Imagem {file_name} não encontrada")
        continue
    
    img = Image.open(img_path)
    
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    if (w*h < 15000):
        continue
    
    crop = img.crop((x1, y1, x2, y2))
    if crop is None:
        continue
    
    crop_name = f"{image_id}_{ann['id']}.jpg"
    crop.save(os.path.join(output_dir, crop_name))

