
from dataloader_triplet import PSGraphTripletDataset

# sg_geometry_path = '/Users/ericnguyen/Desktop/gcn-cnn-mod/data/test_list_aug_feats.pickle'

sg_geometry_path = '/Users/ericnguyen/Desktop/fingerprinting/data_psbattle/psbattles_pos_neg/feats5/geometry-directed/'
det_feats_path = '/Users/ericnguyen/Desktop/fingerprinting/data_psbattle/psbattles_pos_neg/feats5/detection_feats_by_img.pkl'
batch_size = 16
split = 'train'
use_box_feats = True
use_neg_anchor = True

dataloader = PSGraphTripletDataset(
        sg_geometry_path, 
        det_feats_path, 
        batch_size=batch_size, 
        split=split, 
        use_box_feats=use_box_feats, 
        use_neg_anchor=use_neg_anchor
        )

data = dataloader.get_batch()
# data = dataloader.get_batch()

print('data infos', data['infos'])


data = dataloader.get_batch()
# data = dataloader.get_batch()

print('data infos', data['infos'])


# for k, v in data['sg_data_a'].items():

#     print('k', k)



# print('sg_data_a: ', data['sg_data_a'])


# batch return structure
'''
# Keys: value
    bounds: meta data
    infos: idx, paths for a, p, n
    sg_data_a: dict, with keys:
        visual
        visual_masks
        obj_boxes
        box_feats
        rela_edges
        rela_feats
        rela_masks


    sg_data_p:
    sg_data_n:



'''