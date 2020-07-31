from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pickle
from multiprocessing import Pool, Value
import argparse
ENDIAN = '<f4'



class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def add(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value


def build_geometry_graph(key):
    counter.add(1)
    feats = all_feats[key]
    num_boxes = feats.shape[0]
    edges = []
    relas = []
    
    # if id == '71073':
    #     print('breakpoint')
    
    # loop thru each obj in this img
    for i in range(num_boxes):
        if Directed:
            start = 0
        else:
            start = i
        for j in range(start, num_boxes):
            if i==j:
                continue
            # iou and dist thresholds
#            if feats[i][j][3] < Iou or feats[i][j][6] > Dist:
#                continue
            edges.append([i, j])
            relas.append(feats[i][j])


    # ask about this later!!!  # in case some trouble is met
    # no edges can be made, no objs detected...
    if edges == []:
        print('img problem: ', key)

#         f = open("../data/single_component_images_directed.txt", "a")
#         f.write('%s\n'%(key))
# #        edges.append([0, 1])
# #        relas.append(feats[0][1])
#         edges.append([0, 0])
#         relas.append(feats[0][0])

    edges = np.array(edges)
    relas = np.array(relas)
    graph = {}
    graph['edges'] = edges
    graph['feats'] = relas

    # need to put file path in the name, probably need to create subdir also
    np.save(os.path.join(save_dir_path, key), graph)

    if counter.value % 100 == 0 and counter.value >= 100:
#    if counter.value % 2 == 0:
        print('{} / {}'.format(counter.value, num_images))






# start execution here

# parse args here
parser = argparse.ArgumentParser(description='calculate geometry feats from obj detection output')
parser.add_argument('-d', '--geo-feats-path', default='', help='path to output of detection')
parser.add_argument('-o', '--out-path', default='', help='path to save to')
    
args = parser.parse_args()

Directed = True  # directed or undirected graph
out_dir_name = "geometry-{}directed".format('' if Directed else 'un')
save_dir_path = os.path.join(args.out_path, out_dir_name)

# '../graph_data/geometry_feats-{}directed.pkl'.format('' if Directed else 'un')

if not os.path.exists(save_dir_path):

    print('creating dir: ', save_dir_path)
    os.makedirs(save_dir_path)


counter = Counter()
print("loading geometry features of all box pairs....")
with open(args.geo_feats_path, 'rb') as f:
    all_feats = pickle.load(f)
num_images = len(all_feats)
print("Loaded %d images...." % num_images)

# for name, feats in all_feats.items():

#     print('name: {}, feats shape: {}'.format(name, feats.shape))

# exit()

#%%
p = Pool(20)
print("[INFO] Start")
results = p.map(build_geometry_graph, all_feats.keys())
print("Done")


# try loading to make sure it works
for key in all_feats.keys():

    graph_path = os.path.join(save_dir_path, key)
    graph_path = graph_path + '.npy'

    rela = np.load(graph_path, allow_pickle=True)[()]

    print('rela feats', rela['feats'].shape)
    print('rela edges', rela['edges'].shape)

    # exit()



#for key in all_feats.keys():
#    build_geometry_graph(key)














