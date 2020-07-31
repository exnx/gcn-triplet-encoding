#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:05:38 2019
    - Dataloader for triplet of graph-data.
    - trainset only includes ids that has valid positive-pairs (iou > threshold.) e.g. 0.6
    - randomly sample anchors [same as previous]
    - for selected anchor, find an positive from positive set (randomly choose if multiple exits)
    - To find the negative,
        1) randomly choose any image except from the pos list
        2) only choose images whose iou is beteen some range (l_iou, h_iou) --> (0.2-0.4)
           The higher the iou the harder is the negative.
        3) Only choose hard examples just below the postive threshold, and above some iou e.g. (0.4-0.7))
@author: dipu
"""

import torch
from torch.utils.data import Dataset
import torch.utils.data as data
import os
from PIL import Image

from torchvision import transforms
import numpy as np
import random
import pickle
import torch.nn.functional as F
from collections import defaultdict
import random

def default_loader(path):
    return Image.open(path).convert('RGB')

def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)
        print('Saved to {}.'.format(fname))

def pickle_load(fname):
    with open(fname, 'rb') as pf:
         data = pickle.load(pf)
         print('Loaded {}.'.format(fname))
         return data


#%%
class PSGraphTripletDataset(Dataset):
    
    def default_loader(path):
        return Image.open(path).convert('RGB')
    
    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, if_shuffle = (split=='train'))
        self.iterators[split] = 0
   
    def __init__(self, opt, transform, batch_size, sg_geometry_path, detection_feats_path, img_dir, split):

        self.batch_size = batch_size
        self.transform = transform
        self.loader = default_loader  # for img loading, using PIL
        self.geometry_relation = True

        self.img_dir = img_dir  # for imgs

        # setup retrieving det feats
        self.detection_feats_path = detection_feats_path  # det feats
        # # detection feats, keys = paths (so need to convert idx to path)
        self.det_feats = pickle.load(open(detection_feats_path, 'rb'))
        
        ## to get det feats, use like this:  self.det_feats[self.idx2path[idx]]

        # setup getting built geo graph
        self.sg_geometry_dir = sg_geometry_path  # geo graph built
        print('\nLoading geometric graphs and features from dir: {}\n'.format(self.sg_geometry_dir))
        self.geom_feat_size = 8

        # create a dict for consistency on legacy code, where they split in one dataloader
        self.idx2path = {split: list(self.det_feats.keys())}
      
        # ** I have another way of getting apn.  He uses in just get_pairs(),

        self.iterators = {split: 0}
        
        for split in self.idx2path.keys():
            print('Assigned %d images to split %s'%(len(self.idx2path[split]), split))
        
        self._prefetch_process = {} # The two prefetch process 
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train', num_workers=4)

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)


    def __len__(self):
        ''' Used by the _prefetch_process() to work properly '''
        return len(self.det_feats)
    

    def __getitem__(self, index):

        ''' 
        Used by the _prefetch_process() to work properly
        '''
#        ix = index #self.split_ix[index]
        
        sg_data = self.get_graph_data_by_idx(index)   
        return (sg_data, index)  

    def _get_geo_graph_by_idx(self, index):

        # get the geo graph data
        sg_geo_dir = self.sg_geometry_dir
        sample_rel_path = self.idx2path[index]  # relative path, used for key look up
        sg_geo_path = os.path.join(sg_geo_dir, sample_name) + '.npy'  # need to add ext
        rela = np.load(sg_geo_path, allow_pickle=True)[()] # dict contains keys of edges and feats

        return rela

    def _get_detection_feats(self, index):

        path = self.idx2path[index]  # relative path, used for key look up
        det_feats = self.det_feats[path]

        return det_feats

    def get_graph_data_by_idx(self, index):

        # retieve geo graph
        rela = self._get_geo_graph_by_idx(index)
        
        # retrieve visual feats
        det_feats = self._get_detection_feats(index)
        visual_feats = det_feats['visual']

        if self.opt.use_box_feats:
            box_feats = self.get_box_feats(det_feats)
            sg_data = {'visual': visual_feats, 'box_feats': box_feats, 'rela': rela, 'box': box}
        else:
            sg_data = {'visual': visual_feats, 'rela': rela, 'box':box}
    
        return sg_data


    def get_box_feats(self, det_feats):

        # full image shape
        H, W, _ = det_feats['hwc']
        
        x1, y1, x2, y2 = np.hsplit(det_feats['box'], 4)

        w = x2-x1
        h = y2-y1
        
        box_feats = np.hstack((0.5 * (x1 + x2) / W, 0.5 * (y1 + y2) / H, w/W, h/H, w*h/(W*H)))
        #box_feats = box_feat / np.linalg.norm(box_feats, 2, 1, keepdims=True)
        return box_feats
    
    
    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size
        sg_batch_a = []
        sg_batch_p = []
        sg_batch_n = []
        

#        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'float32')
        infos = []
        
        images_a = []
        images_p = []
        images_n = []
        
        wrapped = False  # represents one full epoch use of the dataloader
        
        for i in range(batch_size):

            # fetches a sample randomly (handles multiple epochs by shuffling)
            tmp_sg_a, idx_a, tmp_wrapped = self._prefetch_process[split].get()

            # based on what is returned, prepare the triplet
            

            
            # ***** probably here is where I replace with my own way of finding apn


            # id_p, id_n, iou_p, iou_p_norm, iou_n, iou_n_norm  = self.get_pairs(ix_a)
            id_p, id_n,  iou_p_norm, iou_n_norm = self.get_pairs(ix_a)
            

            

            tmp_sg_p, tmp_img_p, ix_p = self.get_graph_data_by_id(id_p)
            tmp_sg_n, tmp_img_n, ix_n = self.get_graph_data_by_id(id_n)
             
            sg_batch_a.append(tmp_sg_a)
            images_a.append(tmp_img_a) 
            
            sg_batch_p.append(tmp_sg_p)
            images_p.append(tmp_img_p)
            
            sg_batch_n.append(tmp_sg_n)
            images_n.append(tmp_img_n) 
            
          
            
           # record associated info as well
            info_dict = {}
            info_dict['ix_a'] = ix_a
            info_dict['id_a'] = self.info[ix_a]['id']
            info_dict['id_p'] = id_p
            info_dict['id_n'] = id_n
            #info_dict['iou_p'] = iou_p
            #info_dict['iou_n'] = iou_n
            info_dict['iou_p_norm'] = iou_p_norm
            info_dict['iou_n_norm'] = iou_n_norm
            
            infos.append(info_dict)
            
            if tmp_wrapped:
                wrapped = True
                break
            
        data = {}
#        max_box_len = max([_.shape[0] for _ in sg_batch['obj']])
        max_box_len_a = max([_['obj'].shape[0] for _ in sg_batch_a])
        max_box_len_p = max([_['obj'].shape[0] for _ in sg_batch_p])
        max_box_len_n = max([_['obj'].shape[0] for _ in sg_batch_n])
        
        # what the heck?
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        data['sg_data_a'] = self.batch_sg(sg_batch_a, max_box_len_a)
        data['sg_data_p'] = self.batch_sg(sg_batch_p, max_box_len_p)
        data['sg_data_n'] = self.batch_sg(sg_batch_n, max_box_len_n)
        
        return data

    def batch_sg(self, sg_batch, max_box_len):
        "batching object, attribute, and relationship data"
        obj_batch = [_['obj'] for _ in sg_batch]
        rela_batch = [_['rela'] for _ in sg_batch]
        box_batch = [_['box'] for _ in sg_batch]
        
        sg_data = {}

        # obj labels, shape: (B, No, 1)
#        sg_data['obj_labels'] = np.zeros([len(obj_batch), max_att_len, self.opt.num_obj_label_use], dtype = 'int')
        sg_data['obj_labels'] = np.zeros([len(obj_batch), max_box_len, 1], dtype = 'int')
        for i in range(len(obj_batch)):
            sg_data['obj_labels'][i, :obj_batch[i].shape[0]] = obj_batch[i]
        
        sg_data['obj_masks'] = np.zeros([len(obj_batch), max_box_len], dtype ='float32')
        for i in range(len(obj_batch)):
            sg_data['obj_masks'][i, :obj_batch[i].shape[0]] = 1
            
            
        sg_data['obj_boxes'] = np.zeros([len(box_batch), max_box_len, 4], dtype = 'float32')
        for i in range(len(box_batch)):
            sg_data['obj_boxes'][i, :len(box_batch[i])] = box_batch[i]    
        
        
        if self.opt.use_box_feats:
            box_feats_batch = [_['box_feats'] for _ in sg_batch]
            sg_data['box_feats'] = np.zeros([len(box_feats_batch), max_box_len, 5], dtype = 'float32')
            for i in range(len(box_feats_batch)):
                sg_data['box_feats'][i, :len(box_feats_batch[i])] = box_feats_batch[i]   
            
        # rela
        max_rela_len = max([_['edges'].shape[0] for _ in rela_batch])
        sg_data['rela_edges'] = np.zeros([len(rela_batch), max_rela_len, 2], dtype = 'int')
        
        if self.geometry_relation:
            sg_data['rela_feats'] = np.zeros([len(rela_batch), max_rela_len, self.geom_feat_size], dtype = 'float32')
        else:
            sg_data['rela_feats'] = np.zeros([len(rela_batch), max_rela_len], dtype='int')
       
        # rela_masks, because not all items in rela_edges and rela_feats are meaningful
        sg_data['rela_masks'] = np.zeros(sg_data['rela_edges'].shape[:2], dtype='float32')

        for i in range(len(rela_batch)):
            sg_data['rela_edges'][i, :rela_batch[i]['edges'].shape[0]] = rela_batch[i]['edges']
            sg_data['rela_feats'][i, :rela_batch[i]['edges'].shape[0]] = rela_batch[i]['feats']
            sg_data['rela_masks'][i, :rela_batch[i]['edges'].shape[0]] = 1

        return sg_data
    
#%%
class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False, num_workers = 4):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
#        self.opt =opt
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle
        self.num_workers = num_workers

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=SubsetSampler(self.dataloader.idx2path[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers= self.num_workers,#1, # 4 is usually enough
                                            worker_init_fn=None,
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.idx2path[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.idx2path[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.idx2path[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[-1] == ix, "ix not equal"

        return tmp + [wrapped]
    
    
#%%
class SubsetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
        #return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)