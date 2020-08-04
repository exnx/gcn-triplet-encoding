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
    
    def reset_iterator(self):
        del self._prefetch_process
        self._prefetch_process = BlobFetcher(self, if_shuffle = (self.split=='train'))
        # self._prefetch_process = BlobFetcher(self, if_shuffle=True)
        self.iterator = 0
   
    def __init__(self, sg_geometry_path, det_feats_path, batch_size=16, split='train', use_box_feats=True, use_neg_anchor=True):

        random.seed(7)

        self.split = split
        self.batch_size = batch_size
        self.loader = default_loader  # for img loading, using PIL
        self.geometry_relation = True
        self.use_box_feats = use_box_feats
        self.use_neg_anchor = use_neg_anchor  # whether to sample negs as anchors too

        # setup retrieving det feats
        self.det_feats_path = det_feats_path  # det feats
        # # detection feats, keys = paths (so need to convert idx to path)

        # use detection feats organized by img, which is output from cal_geo_feats
        # dict of dicts
        self.det_feats = pickle.load(open(self.det_feats_path, 'rb'))



        # setup getting built geo graph
        self.sg_geometry_dir = sg_geometry_path  # geo graph built
        print('\nLoading detection feats from dir: {}\n'.format(self.det_feats_path))
        print('\nLoading geo graphs from dir: {}\n'.format(self.sg_geometry_dir))
        self.geom_feat_size = 8

        # get paths, put in list
        self.idx2path = list(self.det_feats.keys())
        self.path2idx = self.get_idx_by_path()  # a dict

        self.idxs = [i for i in range(len(self.idx2path))]
      
        # ** I have another way of getting apn.  He uses in just get_pairs(),

        self.iterator = 0
        
        print('Assigned %d images to split %s'%(len(self.idx2path), split))
        
        # only shuffle if train set
        self._prefetch_process = BlobFetcher(self, self.split=='train', num_workers=4)

        def cleanup():
            print('Terminating BlobFetcher')
            del self._prefetch_process
        import atexit
        atexit.register(cleanup)


    def __len__(self):
        ''' Used by the _prefetch_process() to work properly '''
        return len(self.det_feats)
    

    def __getitem__(self, index):

        ''' 
        Used by the _prefetch_process() to work properly
        '''

        # print('get item called {}:'.format(index))

        sg_data = self.get_full_graph_by_idx(index)   
        return [sg_data, index]  # does it have to be a tuple?


    def get_idx_by_path(self):

        ''' key = path, val = idx '''

        idx_by_path = {}

        for idx, path in enumerate(self.idx2path):

            idx_by_path[path] = idx

        return idx_by_path


    def _get_geo_graph_by_idx(self, index):

        # get the geo graph data
        sg_geo_dir = self.sg_geometry_dir
        sample_rel_path = self.idx2path[index]  # relative path, used for key look up
        sg_geo_path = os.path.join(sg_geo_dir, sample_rel_path) + '.npy'  # need to add ext
        rela = np.load(sg_geo_path, allow_pickle=True)[()] # dict contains keys of edges and feats

        return rela

    def _get_det_feats_by_idx(self, index):

        path = self.idx2path[index]  # relative path, used for key look up
        det_feats = self.det_feats[path]

        return det_feats

    def get_full_graph_by_idx(self, index):

        # retieve geo graph
        rela = self._get_geo_graph_by_idx(index)
        
        # retrieve visual feats
        det_feats = self._get_det_feats_by_idx(index)

        visual_feats = det_feats['visual']
        box = det_feats['box']  # in xyxy format, changed from xywh format
        obj_count = det_feats['obj_count']

        obj_count = np.reshape(obj_count, (-1, 1))  # make a N x 1 vector

        # print('obj_count reshaped', obj_count)

        if self.use_box_feats:
            box_feats = self.get_box_feats(det_feats)
            sg_data = {'visual': visual_feats, 'box_feats': box_feats, 'rela': rela, 'box': box}
        else:
            sg_data = {'visual': visual_feats, 'rela': rela, 'box':box}
    
        return sg_data


    def get_box_feats(self, det_feats):

        ''' uses the detection feats to calc the box feats'''

        # full image shape
        H, W, _ = det_feats['hwc'].astype(float)
        
        x1, y1, x2, y2 = np.hsplit(det_feats['box'].astype(float), 4)

        w = x2-x1
        h = y2-y1
        
        box_feats = np.hstack((0.5 * (x1 + x2) / W, 0.5 * (y1 + y2) / H, w/W, h/H, w*h/(W*H)))
        #box_feats = box_feat / np.linalg.norm(box_feats, 2, 1, keepdims=True)
        return box_feats

    def get_triplet(self, q_path):

        ''' 
        
        Given path of a query, build the triplet

        Labels are in the path, which will tell us how to choose negs and pos 
        format:  0000/pos_2kaq0r_00.jpg, -> subdir/pos_img_id.ext, or subdir/neg_img_id.ext
        
        '''

        img_nums_per_label = 4  # per label

        # parse q path
        q_subdir, q_file = q_path.split('/')
        q_label, q_img_name, q_img_num = q_file.split('_')
        q_img_num = int(q_img_num)


        # set anchor positive no matter what
        if not self.use_neg_anchor:
            if q_label == 'neg':

                n_path = q_path  # neg is the anchor

                # sample 2 diff numbers for img nums to be a and p
                a_num, p_num = np.random.choice(img_nums_per_label, 2, replace=False)

                # build up anchor path
                a_file_name = '_'.join(['pos', q_img_name, str(a_num).zfill(2)])
                a_path = os.path.join(q_subdir, a_file_name)

                # build up pos path
                p_file_name = '_'.join(['pos', q_img_name, str(p_num).zfill(2)])
                p_path = os.path.join(q_subdir, p_file_name)

            else: # q label is pos

                a_path = q_path

                # find a different pos
                nums = [i for i in range(img_nums_per_label)] 

                n_num = np.random.choice(nums)  # can be same as query num, since diff label

                # # sample a diff number for img nums to be p, exclude the query
                # p_num = np.random.choice(nums.remove(q_img_num))

                nums.remove(q_img_num)

                p_num = np.random.choice(nums)
                
                
                # build up anchor path
                p_file_name = '_'.join(['pos', q_img_name, str(p_num).zfill(2)])
                p_path = os.path.join(q_subdir, p_file_name)

                # build up pos path
                n_file_name = '_'.join(['neg', q_img_name, str(n_num).zfill(2)])
                n_path = os.path.join(q_subdir, n_file_name)

        # allow neg anchor
        else:

            # same = is same label as anchor
            # diff = diff label than anchor
            # num_idx = 0-49

            # just use whatever the query path is as the anchor (neg or pos)
            a_path = q_path

            # selecting a random num idx
            nums = [i for i in range(img_nums_per_label)]
            diff_num = np.random.choice(nums)
            nums.remove(q_img_num)
            same_num = np.random.choice(nums)

            # get the right name
            if q_label == 'pos':
                same_label = 'pos'
                diff_label = 'neg'
            else:
                diff_label = 'pos'
                same_label = 'neg'

            # build up same path
            same_file_name = '_'.join([same_label, q_img_name, str(same_num).zfill(2)])
            same_path = os.path.join(q_subdir, same_file_name)

            # build up diff path
            diff_file_name = '_'.join([diff_label, q_img_name, str(diff_num).zfill(2)])
            diff_path = os.path.join(q_subdir, diff_file_name)

            p_path = same_path
            n_path = diff_path

        return a_path, p_path, n_path
    
    
    def get_batch(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        sg_batch_a = []
        sg_batch_p = []
        sg_batch_n = []
        
        infos = []
        
        wrapped = False  # represents one full epoch use of the dataloader
        
        for i in range(batch_size):

            # fetches a sample randomly, can be a photoshop version as anchor. handles multiple epochs by shuffling
            tmp_sg_a, idx_a, tmp_wrapped = self._prefetch_process.get()

            path_a = self.idx2path[idx_a]

            # based on what is returned, find the pos and neg versions
            a_path, p_path, n_path = self.get_triplet(path_a)
            a_idx, p_idx, n_idx = self.path2idx[a_path], self.path2idx[p_path], self.path2idx[n_path]

            # retrieve the graphs
            graph_a = self.get_full_graph_by_idx(a_idx)
            graph_p = self.get_full_graph_by_idx(p_idx)
            graph_n = self.get_full_graph_by_idx(n_idx)

            sg_batch_a.append(graph_a)
            sg_batch_p.append(graph_p)    
            sg_batch_n.append(graph_n)
            
           # record associated info as well
            info_dict = {}
            info_dict['ix_a'] = idx_a
            info_dict['a_path'] = a_path
            info_dict['p_path'] = p_path
            info_dict['n_path'] = n_path
            
            infos.append(info_dict)
            
            if tmp_wrapped:
                wrapped = True
                break
            
        data = {}

        # find max number of objects in an image across each batch
        max_box_len_a = max([_['visual'].shape[0] for _ in sg_batch_a])
        max_box_len_p = max([_['visual'].shape[0] for _ in sg_batch_p])
        max_box_len_n = max([_['visual'].shape[0] for _ in sg_batch_n])

        # print('max_box_len_a: ', max_box_len_a)
        # print('max_box_len_p: ', max_box_len_p)
        # print('max_box_len_n: ', max_box_len_n)

        # just meta data on the batch
        data['bounds'] = {'it_pos_now': self.iterator, 'it_max': len(self.idx2path), 'wrapped': wrapped}
        data['infos'] = infos  # a/p/n paths and idxs used in batch

        data['sg_data_a'] = self.batch_sg(sg_batch_a, max_box_len_a)
        data['sg_data_p'] = self.batch_sg(sg_batch_p, max_box_len_p)
        data['sg_data_n'] = self.batch_sg(sg_batch_n, max_box_len_n)
        
        return data

    def batch_sg(self, sg_batch, max_box_len):
        "batching object, attribute, and relationship data"

        rela_batch = [_['rela'] for _ in sg_batch]
        box_batch = [_['box'] for _ in sg_batch]
        visual_batch = [_['visual'] for _ in sg_batch]

        sg_data = {}

        # visual feats, shape: (B, max_box_len, 128) 
        sg_data['visual'] = np.zeros([len(visual_batch), max_box_len, visual_batch[0].shape[1]], dtype = 'float32')
        for i in range(len(visual_batch)):
            # for ith sample, set all up to its num of objects
            sg_data['visual'][i, :visual_batch[i].shape[0]] = visual_batch[i]

        sg_data['visual_masks'] = np.zeros(sg_data['visual'].shape[:2], dtype='float32')
        for i in range(len(visual_batch)):
            sg_data['visual_masks'][i, :visual_batch[i].shape[0]] = 1


        sg_data['obj_boxes'] = np.zeros([len(box_batch), max_box_len, 4], dtype = 'float32')
        for i in range(len(box_batch)):
            sg_data['obj_boxes'][i, :len(box_batch[i])] = box_batch[i]
        
        if self.use_box_feats:
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
    def __init__(self, dataloader, if_shuffle=False, num_workers = 4):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
#        self.opt =opt
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle
        self.num_workers = num_workers

        # we need the first epoch to be shuffled, before it was in order to annotations
        # even for val set
        random.shuffle(self.dataloader.idxs)  

        # self.reset()  # shuffle at beginning once

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
                                            sampler=SubsetSampler(self.dataloader.idxs[self.dataloader.iterator:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers= self.num_workers,#1, # 4 is usually enough
                                            worker_init_fn=None,
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.idxs)
        wrapped = False

        ri = self.dataloader.iterator
        ix = self.dataloader.idxs[ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                print('shuffling')
                random.shuffle(self.dataloader.idxs)
            wrapped = True
        self.dataloader.iterator = ri_next

        return ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        try:
            assert tmp[-1] == ix, "ix not equal"
        except Exception as E:
            print('ix {}, tmp[-1 {}'.format(ix, tmp[-1]))
            print(E)

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
        # print('randperm')
        return (self.indices[i] for i in range(len(self.indices)))
        # return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)