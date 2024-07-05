
import os
import csv

os.environ["DGLBACKEND"] = "pytorch"
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

def convert_numpy_to_dict(np_array):
    temp_dict = {}
    for i in range(np_array[:,0].size):
        temp_idx = np_array[i][0]
        temp_field = np_array[i][1]
        temp_value = np_array[i][2]

        if (temp_field, temp_value) in temp_dict.keys():
            print("already exist...")
        else:
            temp_dict[(temp_field, temp_value)] = temp_idx
    
    print(temp_dict)
    return temp_dict


print("the dataset...")
file_location = "../../../dataset"
dataset_name = 'ciao'
print('Enter the exact folder name:')
res_dir = input()
print('Enter gnn model name:')
modelname = input()

res_filename =  'makex_result/' + modelname + '/' + res_dir + '/'
up_dir_to_load = file_location + "/" + dataset_name + "/" + res_filename

sub_dirs = os.listdir(up_dir_to_load)

for sub_dir in sub_dirs:
    print(sub_dir)
    dir_to_load = up_dir_to_load + sub_dir + '/'

    if os.path.isdir(dir_to_load) == False:
        continue

    attr_filename = "/v.csv"

    dir_to_load_attr = file_location + "/" + dataset_name
    attr_map_filename = "/attribute_to_embedding_mapping.csv"
    data_types = {
        'pair_id': 'Int64',
        'pivot_x': 'Int64',
        'pivot_y': 'Int64',
        'topk': 'Int64',
        'vertex_id': 'Int64',
        'label_id:int': 'Int64',
        'genres:int': 'Int64'
    }

    attributes = pd.read_csv(dir_to_load + attr_filename, dtype=data_types)

    attr_mapping =  pd.read_csv(dir_to_load_attr + attr_map_filename, header=None).to_numpy()
    print(attr_mapping[0])
    cnt_values_in_attr = len(attr_mapping)
    print("attr mapping has {} values".format(cnt_values_in_attr))

    dim_attr = 16
    dim_node = 16

    fields_name = ['genres']
    genres = attributes['genres:int'].to_numpy()

    attributes = genres
    print(attributes.shape)

    attr_mapping_dict = convert_numpy_to_dict(attr_mapping)
    embedding_lookup_table = nn.Embedding(cnt_values_in_attr + 2, dim_attr) 

    ids_of_feature_vecs = [] 
    feature_for_lookup = [] 

    for i in range(attributes.shape[0]):

        temp_embed_lookup_ids = None
        temp_pair = attributes[i]

        temp_field = fields_name[0]
        temp_value = temp_pair
        if pd.isna(temp_value) or temp_value == 'nan':
            temp_embed_lookup_ids = cnt_values_in_attr + 1
        else:
            mapped_idx = attr_mapping_dict[(temp_field, int(temp_value))]
            temp_embed_lookup_ids= mapped_idx
        
            
        ids_of_feature_vecs.append(i)
        feature_for_lookup.append([temp_embed_lookup_ids])

    temp_input = torch.LongTensor(feature_for_lookup)
    print(temp_input.shape)
    temp_vecs = embedding_lookup_table(temp_input)
    print(temp_vecs.shape)
    temp_x_dims = temp_vecs.shape[0]
    temp_y_dims = temp_vecs.shape[1] * temp_vecs.shape[2]

    temp_vecs = temp_vecs.reshape([temp_x_dims, temp_y_dims])
    print(temp_vecs.shape)

    connected_layer = nn.Linear(temp_vecs.shape[1], dim_node)
    print(connected_layer)

    final_feature_vector = connected_layer(temp_vecs)

    print(final_feature_vector.shape)

    feature_vector = final_feature_vector.cpu().detach().numpy()

    feat_vecs_filename = dir_to_load + "/feature_vectors_v.csv"
    vec_file_df = pd.DataFrame(feature_vector)
    vec_file_df.to_csv(feat_vecs_filename)