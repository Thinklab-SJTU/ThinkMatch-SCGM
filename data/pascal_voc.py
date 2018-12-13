from pathlib import Path
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import random
import pickle
import torch
import sys

from utils.config import cfg

anno_path = cfg.VOC2011.KPT_ANNO_DIR
img_path = cfg.VOC2011.ROOT_DIR + 'JPEGImages'
ori_anno_path = cfg.VOC2011.ROOT_DIR + 'Annotations'
set_path = cfg.VOC2011.SET_SPLIT
cache_path = cfg.CACHE_PATH

#random.seed(0)


class PascalVOC:
    def __init__(self, sets, obj_resize):
        """
        :param sets: 'train' or 'test'
        :param obj_resize: resized object size
        """
        self.classes = cfg.VOC2011.CLASSES

        self.anno_path = Path(anno_path)
        self.img_path = Path(img_path)
        self.ori_anno_path = Path(ori_anno_path)
        self.obj_resize = obj_resize

        assert sets == 'train' or 'test', 'No match found for dataset {}'.format(sets)
        cache_name = 'voc_db_' + sets + '.pkl'
        self.cache_path = Path(cache_path)
        self.cache_file = self.cache_path / cache_name
        if self.cache_file.exists():
            with self.cache_file.open(mode='rb') as f:
                self.xml_list = pickle.load(f)
            print('xml list loaded from {}'.format(self.cache_file))
        else:
            print('Caching xml list to {}...'.format(self.cache_file))
            self.cache_path.mkdir(exist_ok=True, parents=True)
            with np.load(set_path) as f:
                self.xml_list = f[sets]
            before_filter = sum([len(k) for k in self.xml_list])
            self.filter_list()
            after_filter = sum([len(k) for k in self.xml_list])
            with self.cache_file.open(mode='wb') as f:
                pickle.dump(self.xml_list, f)
            print('Filtered {} images to {}. Annotation saved.'.format(before_filter, after_filter))

    def filter_list(self):
        """
        Filter out 'truncated', 'occluded' and 'difficult' images following the practice of previous works.
        In addition, this dataset has uncleaned label (in person category). They are omitted as suggested by README.
        """
        for cls_id in range(len(self.classes)):
            to_del = []
            for xml_name in self.xml_list[cls_id]:
                xml_comps = xml_name.split('/')[1].strip('.xml').split('_')
                ori_xml_name = '_'.join(xml_comps[:-1]) + '.xml'
                voc_idx = int(xml_comps[-1])
                xml_file = self.ori_anno_path / ori_xml_name
                assert xml_file.exists(), '{} does not exist.'.format(xml_file)
                tree = ET.parse(xml_file.open())
                root = tree.getroot()
                obj = root.findall('object')[voc_idx - 1]

                difficult = obj.find('difficult')
                if difficult is not None: difficult = int(difficult.text)
                occluded = obj.find('occluded')
                if occluded is not None: occluded = int(occluded.text)
                truncated = obj.find('truncated')
                if truncated is not None: truncated = int(truncated.text)
                if difficult or occluded or truncated:
                    to_del.append(xml_name)
                    continue

                # Exclude uncleaned images
                if self.classes[cls_id] == 'person' and int(xml_comps[0]) > 2008:
                    to_del.append(xml_name)
                    continue

            for x in to_del:
                self.xml_list[cls_id].remove(x)

    def get_pair(self, cls=None):
        """
        Randomly get a pair of objects from VOC-Berkeley keypoints dataset
        :param idx: index, used for random seed
        :param cls: None for random class, or specify for a certain set
        :return: (pair of data, groundtruth permutation matrix)
        """
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        anno_pair = []
        for xml_name in random.sample(self.xml_list[cls], 2):
            xml_file = self.anno_path / xml_name
            assert xml_file.exists(), '{} does not exist.'.format(xml_file)

            tree = ET.parse(xml_file.open())
            root = tree.getroot()

            img_name = root.find('./image').text + '.jpg'
            img_file = self.img_path / img_name
            bounds = root.find('./visible_bounds').attrib
            h = float(bounds['height'])
            w = float(bounds['width'])
            xmin = float(bounds['xmin'])
            ymin = float(bounds['ymin'])
            with Image.open(str(img_file)) as img:
                ori_sizes = img.size
                obj = img.resize(self.obj_resize, resample=Image.BICUBIC, box=(xmin, ymin, xmin + w, ymin + h))

            keypoint_list = []
            for keypoint in root.findall('./keypoints/keypoint'):
                attr = keypoint.attrib
                attr['x'] = (float(attr['x']) - xmin) * self.obj_resize[0] / w
                attr['y'] = (float(attr['y']) - ymin) * self.obj_resize[1] / h
                keypoint_list.append(attr)

            anno_dict = dict()
            anno_dict['image'] = obj
            anno_dict['keypoints'] = keypoint_list
            anno_dict['bounds'] = xmin, ymin, w, h
            anno_dict['ori_sizes'] = ori_sizes
            anno_dict['cls'] = cls

            anno_pair.append(anno_dict)

        perm_mat = np.zeros([len(_['keypoints']) for _ in anno_pair], dtype=np.float32)
        row_list = []
        col_list = []
        for i, keypoint in enumerate(anno_pair[0]['keypoints']):
            for j, _keypoint in enumerate(anno_pair[1]['keypoints']):
                if keypoint['name'] == _keypoint['name']:
                    perm_mat[i, j] = 1
                    row_list.append(i)
                    col_list.append(j)
                    break
        perm_mat = perm_mat[row_list, :]
        perm_mat = perm_mat[:, col_list]
        anno_pair[0]['keypoints'] = [anno_pair[0]['keypoints'][i] for i in row_list]
        anno_pair[1]['keypoints'] = [anno_pair[1]['keypoints'][j] for j in col_list]

        return anno_pair, perm_mat


if __name__ == '__main__':
    dataset = PascalVOC('train', (256, 256))
    a = dataset.get_pair()
    pass
