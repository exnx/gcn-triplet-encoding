from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import pickle
from multiprocessing import Pool, Value
from collections import defaultdict
#import dill
import argparse
import os
import time


'''

Reads a pickle of features from object detection and extractor, then will
create geometric relationship features.

input:
    .pickle file, from detection output

Return is a dictionary, 
    keys = str, img path
    val = np arr, img geo rela feats
    size n x n x 8, where n is the num of objects in that img, and
    8 is the feat dimensions


'''

class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def add(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value


def get_cwh(box):
    x_min, y_min, x_max, y_max = box
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.  # why plus 1?
    h = (y_max - y_min) + 1.
    return cx, cy, w, h


def convert_xywh_to_xyxy(box):
    x_min, y_min, w, h = box
    x_max = x_min + w
    y_max = y_min + h
    
#    if w <1:
#        w=1
#    if h <1:
#        h=1    
    return([x_min, y_min, x_max, y_max])

#def convert(old_pkl):
#    import os 
#    """
#    Convert a Python 2 pickle to Python 3
#    """
#    # Make a name for the new pickle
#    new_pkl = os.path.splitext(os.path.basename(old_pkl))[0]+"_p3.pkl"
#
#    # Convert Python 2 "ObjectType" to Python 3 object
#    dill._dill._reverse_typemap["ObjectType"] = object
#
#    # Open the pickle using latin1 encoding
#    with open(old_pkl, "rb") as f:
#        loaded = pickle.load(f, encoding="latin1")
#
#    # Re-save as Python 3 pickle
#    with open(new_pkl, "wb") as outfile:
#        pickle.dump(loaded, outfile)



def cal_geometry_feats(key):
    counter.add(1)
    info = box_info_dicts[key]
    # boxes = info['xywh']
    boxes = info['box']
    locs = info['loc']
    num_boxes = info['obj_count']

    # shape of whole image
    h, w, c = info['hwc'].astype(float)
    
    # w, h = 1440, 2560
    scale = w * h

    diag_len = math.sqrt(w ** 2 + h ** 2)
    
    feats = np.zeros([num_boxes, num_boxes, NumFeats], dtype='float')
    
    for i in range(num_boxes):

        if Directed:
            start = 0
        else:
            start = i
        
        for j in range(start, num_boxes):

            # boxes already in xyxy format     
            box1, box2 = boxes[i], boxes[j]

            # #Convet to xyxy format, as it is saved as xywh
            # box1 = convert_xywh_to_xyxy(box1)
            # box2 = convert_xywh_to_xyxy(box2)
            
            cx1, cy1, w1, h1 = get_cwh(box1)
            cx2, cy2, w2, h2 = get_cwh(box2)
            
            x_min1, y_min1, x_max1, y_max1 = box1
            x_min2, y_min2, x_max2, y_max2 = box2
            
            # scale
            scale1 = w1 * h1
            scale2 = w2 * h2
            
            # Offset
            offsetx = cx2 - cx1
            offsety = cy2 - cy1
            
            # Aspect ratio
            aspect1 = w1 / h1
            aspect2 = w2 / h2
            
            # Overlap (IoU)
            i_xmin = max(x_min1, x_min2)
            i_ymin = max(y_min1, y_min2)
            i_xmax = min(x_max1, x_max2)
            i_ymax = min(y_max1, y_max2)
            iw = max(i_xmax - i_xmin + 1, 0)
            ih = max(i_ymax - i_ymin + 1, 0)
            areaI = iw * ih
            areaU = scale1 + scale2 - areaI
            
            # dist
            dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            
            # angle
            angle = math.atan2(cy2 - cy1, cx2 - cx1)

            f1 = offsetx / math.sqrt(scale1)
            f2 = offsety / math.sqrt(scale1)     
            f3 = math.sqrt(scale2 / scale1)                
            f4 = areaI / areaU
            f5 = aspect1
            f6 = aspect2
            f7 = dist / diag_len
            f8 = angle
            feat = [f1, f2, f3, f4, f5, f6, f7, f8]
            feats[i][j] = np.array(feat)
    
    if counter.value % 100 == 0 and counter.value >= 100:
        print('{} / {}'.format(counter.value, num_imgs))
    return key, feats


def convert_to_dicts(obj_feats, seq_len):

    """
    trim the number of objects in an image, based on highest semantic scores

    return a dict of dictionaries, 1 for a each image and its feats

    key = path, val = features

    """

    obj_count_by_id = obj_feats['obj_count']
    num_imgs = len(obj_count_by_id)
    print('num of imgs: ', num_imgs)

    # start and end position of samples in data array
    obj_chunks_inds = list(np.cumsum(obj_count_by_id))

    # # for each key, store the feats by id
    all_feats_by_id = {}  # key = feat name, v = list of features by id

    feats_by_img_level = ['obj_count', 'hwc', 'path']

    # split all features into lists searchable by img id, store in a dict
    # if feats already by img, then skip
    for feat_name, feats in obj_feats.items():

        # these feats were organized by obj level
        if feat_name not in feats_by_img_level:

            # group objects by img
            feat_by_img_id = np.split(feats, obj_chunks_inds)[:-1]
            all_feats_by_id[feat_name] = feat_by_img_id

        # these feats were organized by img level
        else:
            all_feats_by_id[feat_name] = feats

    # feats by id
    score_by_id = all_feats_by_id['score']
    path_by_id = all_feats_by_id['path']
    sem_by_id = all_feats_by_id['sem']
    visual_by_id = all_feats_by_id['visual']
    box_by_id = all_feats_by_id['box']
    loc_by_id = all_feats_by_id['loc']
    shape_by_id = all_feats_by_id['shape']
    hwc_by_id = all_feats_by_id['hwc']

    dict_of_img_dicts = {}  # store output here

    # loop thru imgs, then loop thru keys and retrieve the img's values
    # and store in a new dict.  Do for each img
    for img_id in range(num_imgs):

        img_feat_dict = {}

        # need to trim
        if obj_count_by_id[img_id] > seq_len:

            # use the highest ranked object up to seq_len (then reverse)
            sel_id = np.argsort(score_by_id[img_id])[::-1][:seq_len][::-1]

            img_feat_dict['loc'] = loc_by_id[img_id][sel_id]
            img_feat_dict['sem'] = sem_by_id[img_id][sel_id]
            img_feat_dict['visual'] = visual_by_id[img_id][sel_id]
            img_feat_dict['box'] = box_by_id[img_id][sel_id]
            img_feat_dict['shape'] = shape_by_id[img_id][sel_id]
        
        else:
            img_feat_dict['loc'] = loc_by_id[img_id]
            img_feat_dict['sem'] = sem_by_id[img_id]
            img_feat_dict['visual'] = visual_by_id[img_id]
            img_feat_dict['box'] = box_by_id[img_id]
            img_feat_dict['shape'] = shape_by_id[img_id]

        # always stays the same
        img_feat_dict['hwc'] = hwc_by_id[img_id]

        # special case, need to update object count and scores last
        if obj_count_by_id[img_id] > seq_len:
            img_feat_dict['obj_count'] = seq_len
            img_feat_dict['score'] = np.sort(score_by_id[img_id])[::-1][:seq_len][::-1]

        else:
            img_feat_dict['obj_count'] = obj_count_by_id[img_id]
            img_feat_dict['score'] = score_by_id[img_id]

        # store with img path as key, and img dict as value
        img_path = path_by_id[img_id]
        dict_of_img_dicts[img_path] = img_feat_dict

    return dict_of_img_dicts




# start execution here

parser = argparse.ArgumentParser(description='calculate geometry feats from obj detection output')
parser.add_argument('-d', '--det-info-path', default='', help='path to output of detection')
parser.add_argument('-o', '--out-path', default='', help='path to save to')
parser.add_argument('-s', '--seq-len', default=64, type=int, help='max num objects')
    
args = parser.parse_args()


NumFeats = 8
Directed = True

out_path = args.out_path

if not os.path.exists(out_path):
    os.makedirs(out_path)

out_name = 'geometry_feats-{}directed.pkl'.format('' if Directed else 'un')
save_path = os.path.join(out_path, out_name)

with open(args.det_info_path, 'rb') as f:
    box_info = pickle.load(f)  # load

    # convert to dict
    box_info_dicts = convert_to_dicts(box_info, args.seq_len)


num_imgs = len(box_info_dicts)
counter = Counter()

p = Pool(20)
print("[INFO] Start")

results = p.map(cal_geometry_feats, box_info_dicts.keys())

all_feats = {res[0]: res[1] for res in results}

print('len of results: ', len(results))

print("[INFO] Finally %d processed" % len(all_feats))


# save all feat out puts
with open(save_path, 'wb') as f:
    pickle.dump(all_feats, f)
print("saved")






exit()

# after saving, test open
# by now feat is 8 dim out
with open(save_path, 'rb') as f:

    info = pickle.load(f)

    for img_path, feat_np in info.items():

        print('img path: ', img_path)

        # print('feat np:', feat_np)

        # for feat, val in feat_dict.items():

        #     if feat in show:

        #         print('feat:', feat)
        #         print('val:', val)




    # BoxInfo = pickle.load(open(args.det_info_path, 'rb'))

    # BoxInfo = pickle.load(open('../data/rico_box_info.pkl', 'rb'))

    # BoxInfo = dict(BoxInfo)

    # Box.Info.keys() = id of each image
    # then BoxInfo['70003'] = dict
        # keys = 'nComponent', int
        # 'componentLabel', list of str
        # 'class_id', list of int
        # 'xywh', list of lists (len 4)
        # 'iconClass', list of str
        # 'textButtonClass'] list of str

    # output
    # feats = edges between every obj in img (fully connected)
    # id = img id



    # print('box info keys:', BoxInfo.keys())


    # print('id 70003 nComponent:', BoxInfo['70003']['nComponent'])
    # print('id 70003 componentLabel:', BoxInfo['70003']['componentLabel'])
    # print('id 70003 class_id:', BoxInfo['70003']['class_id'])
    # print('id 70003 xywh:', BoxInfo['70003']['xywh'])
    # print('id 70003 iconClass:', BoxInfo['70003']['iconClass'])
    # print('id 70003 textButtonClass:', BoxInfo['70003']['textButtonClass'])



## from largest det. output
# python ~/Desktop/gcn-cnn/graph_scripts/cal_geometry_feat.py \
#     --det-info-path ~/Desktop/gcn-cnn/data/test_list_aug_feats.pickle \
#     --out-path ~/Desktop/gcn-cnn/graph_data/
     
## small test set
# python ~/Desktop/gcn-cnn/graph_scripts/cal_geometry_feat.py \
#     --det-info-path /Users/ericnguyen/Desktop/fingerprinting/data_psbattle/psbattles_pos_neg/000/list_aug_feats.pickle \
#     --out-path ~/Desktop/fingerprinting/data_psbattle/psbattles_pos_neg/000/
     
     
     
     
     
     
     
     
     
     
     
     