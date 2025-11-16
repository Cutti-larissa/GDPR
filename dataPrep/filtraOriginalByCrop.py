import os
import json
import shutil

filtered_dir = "PersonValCrop"
original_images_dir = "PersonValRaw"
json_path = "val_person_with_head.json"
output_originals_dir = "Filtradas"

os.makedirs(output_originals_dir, exist_ok=True)

with open(json_path, "r") as f:
    coco = json.load(f)

id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

image_ids = set()
for file_name in os.listdir(filtered_dir):
    if not file_name.lower().endswith(".jpg"):
        continue
    image_id = file_name.split("_")[0] # nome das imagens filtradas
    image_ids.add(int(image_id))

for img_id in image_ids:
    if img_id not in id_to_filename:
        print(f" ID {img_id} não encontrado no JSON.")
        continue
    original_name = id_to_filename[img_id] # nome das imagens originais a partir das filtradas
    src_path = os.path.join(original_images_dir, original_name)
    dst_path = os.path.join(output_originals_dir, original_name)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
