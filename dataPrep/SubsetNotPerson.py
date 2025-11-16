# Códigos adaptados de: 
# https://stackoverflow.com/questions/60227833/how-to-filter-coco-dataset-classes-annotations-for-custom-dataset

import os
import json
import requests
from tqdm import tqdm
from os.path import join
from pycocotools.coco import COCO

class coco_category_filter:
    def __init__(self, json_path, imgs_dir):
        self.coco = COCO(json_path)
        self.json_path = json_path
        self.imgs_dir = imgs_dir
        self.images = self.get_imgs_from_json()        
     
    def get_imgs_from_json(self):
        print("Selecionando as imagens!")
        notPerson_id = self.coco.getCatIds(catNms=['bicycle', 'bird', 'cat', 'dog', 'bear', 'surfboard', 
                                                    'teddy bear', 'bench', 'horse', 'sheep', 'cow', 'elephant', 
                                                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
                                                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
                                                    'baseball bat', 'baseball glove', 'skateboard',  'surfboard', 
                                                    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 
                                                    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',  
                                                    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                                                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
                                                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
                                                    'toaster', 'sink', 'refrigerator', 'book', 'vase', 'scissors', 'hair drier', 'toothbrush'])
        person_id = self.coco.getCatIds(catNms=['person'])

        img_ids_with_all_cats = set(self.coco.getImgIds(catIds=[notPerson_id[0]]))
        for i in range (1, len(notPerson_id)):
            current_cat_img_ids = set(self.coco.getImgIds(catIds=[notPerson_id[i]]))
            img_ids_with_all_cats = img_ids_with_all_cats.union(current_cat_img_ids) - img_ids_with_all_cats.intersection(current_cat_img_ids)

        notPerson_imgs_ids = list(img_ids_with_all_cats)
        person_imgs_ids = list(self.coco.getImgIds(catIds=person_id))

        valid_imgs_ids = list(filter(lambda item: item not in person_imgs_ids, notPerson_imgs_ids))

        images = self.coco.loadImgs(list(valid_imgs_ids))
        
        self.catIds = [notPerson_id]

        return images
 
    def save_imgs(self):
        print("Salvando as imagens!")
        os.makedirs(self.imgs_dir, exist_ok=True)

        for im in tqdm(self.images):
            img_data = requests.get(im['coco_url']).content
            with open(os.path.join(self.imgs_dir, im['file_name']), 'wb') as handler:
                handler.write(img_data)
 
    def filter_json_by_category(self, new_json_path):
        print("Filtrando as anotações!")
        json_parent = os.path.split(new_json_path)[0]
        os.makedirs(json_parent, exist_ok=True)
        
        imgs_ids = [x['id'] for x in self.images] 
        new_imgs = [x for x in self.coco.dataset['images'] if x['id'] in imgs_ids]
        catIds = self.catIds

        new_annots = [x for x in self.coco.dataset['annotations'] if x['category_id'] in catIds]
        new_imgs, annotations = self.modify_ids(new_imgs, new_annots)
        new_categories = [x for x in self.coco.dataset['categories'] if x['id'] in catIds]
        
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


def main(subset, year, root_dir, category):
    json_file = join(os.path.split(root_dir)[0], 'instances_'+ subset + year +'.json')
    
    imgs_dir = join(root_dir, category.replace(" ", "_") + '_' + subset)
    new_json_file = join(root_dir, 'annotations', subset + category.replace(" ", "_")+".json")
    
    coco_filter = coco_category_filter(json_file, imgs_dir)
    coco_filter.save_imgs()
    coco_filter.filter_json_by_category(new_json_file)


if __name__ == '__main__':
    subset, year = 'train', '2017'
    root_dir = './annotations/'
    main(subset, year, root_dir, category='NotPerson')
