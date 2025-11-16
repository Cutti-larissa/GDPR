# Códigos adaptados de: 
# https://stackoverflow.com/questions/60227833/how-to-filter-coco-dataset-classes-annotations-for-custom-dataset

import os
import json
import requests
from tqdm import tqdm
from os.path import join
from pycocotools.coco import COCO

class coco_person_head_filter:
    def __init__(self, json_path, imgs_dir):
        self.coco = COCO(json_path)
        self.json_path = json_path
        self.imgs_dir = imgs_dir
        self.images, self.annotations = self.get_imgs_with_head()        

    def get_imgs_with_head(self):
        print("Selecionando as imagens!")
        person_id = self.coco.getCatIds(catNms=['person'])[0]
        imgIds = self.coco.getImgIds(catIds=[person_id])

        valid_annotations = []
        valid_imgIds = set()

        for imgId in imgIds:
            annIds = self.coco.getAnnIds(imgIds=imgId, catIds=[person_id])
            anns = self.coco.loadAnns(annIds)
            for ann in anns:
                if 'keypoints' in ann:
                    k = ann['keypoints']

                    nose_v = k[2]
                    left_eye_v = k[5]
                    right_eye_v = k[8]
                    left_ear_v = k[11]
                    right_ear_v = k[14]

                    if any(v > 0 for v in [nose_v, left_eye_v, right_eye_v, left_ear_v, right_ear_v]):
                        valid_annotations.append(ann)
                        valid_imgIds.add(imgId)

        images = self.coco.loadImgs(list(valid_imgIds))

        return images, valid_annotations

    def save_imgs(self):
        print("Saving the images with required categories ...")
        os.makedirs(self.imgs_dir, exist_ok=True)
        for im in tqdm(self.images):
            img_data = requests.get(im['coco_url']).content
            with open(os.path.join(self.imgs_dir, im['file_name']), 'wb') as handler:
                handler.write(img_data)

    def filter_json(self, new_json_path):
        print("Filtrando as anotações!")
        json_parent = os.path.split(new_json_path)[0]
        os.makedirs(json_parent, exist_ok=True)

        imgs_ids = [x['id'] for x in self.images]
        new_imgs = [x for x in self.coco.dataset['images'] if x['id'] in imgs_ids]
        person_id = self.coco.getCatIds(catNms=['person'])[0]

        new_imgs, new_annots = self.modify_ids(new_imgs, self.annotations)
        new_categories = [x for x in self.coco.dataset['categories'] if x['id'] == person_id]

        data = {
            "info": self.coco.dataset['info'],
            "licenses": self.coco.dataset['licenses'],
            "images": new_imgs, 
            "annotations": new_annots,
            "categories": new_categories 
        }

        print("saving json: ")
        with open(new_json_path, 'w') as f:
            json.dump(data, f)

    def modify_ids(self, images, annotations):
        print("Reorganizando os IDs das imagens e anotações")
        old_new_imgs_ids = {}
        for n, im in enumerate(images):
            old_new_imgs_ids[images[n]['id']] = n + 1
            images[n]['id'] = n + 1
        for n, ann in enumerate(annotations):
            annotations[n]['id'] = n + 1
            old_image_id = annotations[n]['image_id']
            annotations[n]['image_id'] = old_new_imgs_ids[old_image_id]
        return images, annotations

def main(subset, year, root_dir):
    json_file = join(os.path.split(root_dir)[0], 'person_keypoints_' + subset + year + '.json')
    
    imgs_dir = join(root_dir, 'person_with_head_' + subset)
    new_json_file = join(root_dir, 'annotations', subset + '_person_with_head.json')

    coco_filter = coco_person_head_filter(json_file, imgs_dir)
    coco_filter.save_imgs()
    coco_filter.filter_json(new_json_file)


if __name__ == '__main__':
    subset, year = 'val', '2017'
    root_dir = './annotations/'
    main(subset, year, root_dir)

