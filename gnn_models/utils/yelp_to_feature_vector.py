import os
import csv

os.environ["DGLBACKEND"] = "pytorch"
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

def count_num_values_in_a_field(filed_list):
    field_dict = {}
    for idx in range(filed_list.size):
        value = filed_list[idx]
        if pd.isna(value):
            continue

        if value in field_dict.keys():
            field_dict[value] = field_dict[value] + 1
        else:
            field_dict[value] = 1

    cnt = len(field_dict)
    print(cnt)

    return field_dict, cnt


def get_attribute_mapping(attributes):
    fields_name = ['postal_code', 'latitude', 'longitude', 'item_review_count', 'is_open', 'user_review_count', 'yelping_since', 'fans', 'average_stars', 'name']
    dict_of_fields = []
    num_values_in_fields = []
    total_num_values = 0
    for idx in range(len(fields_name)):
        new_idx = idx + 2
        field_array = attributes[:,new_idx]
        temp_dict, temp_cnt = count_num_values_in_a_field(field_array)

        dict_of_fields.append(temp_dict)
        num_values_in_fields.append(temp_cnt)
        total_num_values = total_num_values + temp_cnt

    print(total_num_values)
    mapping_filename = dir_to_load + "/attribute_to_embedding_mapping.csv"
    with open(mapping_filename, 'w') as map_file:
        map_writer = csv.DictWriter(map_file, fieldnames=['idx','field_name', 'value'])
        true_idx = 0
        for i in range(len(dict_of_fields)):
            temp_dict = dict_of_fields[i]
            for temp_k, temp_v in temp_dict.items():
                temp_map_dict = {
                    "idx": true_idx,
                    "field_name": fields_name[i],
                    "value": temp_k
                }
                map_writer.writerow(temp_map_dict)
                true_idx = true_idx + 1

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
    
    return temp_dict

print("start processing the dataset...")
file_location = "../../dataset/"
dataset_name = "yelp"
dir_to_load = file_location + dataset_name
attr_filename = "/new_attributes_v.csv"
attr_map_filename = "/attribute_to_embedding_mapping.csv"

dtypes = {
    "vertex_id": int,
    "label_id": int,
    "postal_code": str,
    "latitude": float,
    "longitude": float,
    "item_review_count": int,
    "is_open": int,
    "user_review_count": int,
    "yelping_since": str,
    "fans": int,
    "average_stars": float,
    "name":str
}
attributes = pd.read_csv(dir_to_load + attr_filename, low_memory=False).to_numpy()
get_attribute_mapping(attributes=attributes)
attr_mapping =  pd.read_csv(dir_to_load + attr_map_filename, header=None).to_numpy()

cnt_values_in_attr = attr_mapping.size

dim_attr = 16
dim_node = 16
fields_name = ['postal_code', 'latitude', 'longitude', 'item_review_count', 'is_open', 'user_review_count', 'yelping_since', 'fans', 'average_stars', 'name']
attr_mapping_dict = convert_numpy_to_dict(attr_mapping)
embedding_lookup_table = nn.Embedding(cnt_values_in_attr, dim_attr)

item_ids_of_feature_vecs = []
item_feature_for_lookup = []

user_ids_of_feature_vecs = []
user_feature_for_loopup = []

for i in range(attributes[:,0].size):
    temp_node = attributes[i]
    temp_node_id = temp_node[0]
    temp_node_label = temp_node[1]
    if temp_node_label > 1:
        break
    
    temp_embed_lookup_ids = []
    for j in range(2, temp_node.size):
        temp_field = fields_name[j-2]
        temp_value = temp_node[j]
        if pd.isna(temp_value):
            continue

        mapped_idx = attr_mapping_dict[(temp_field, str(temp_value))]
        temp_embed_lookup_ids.append(mapped_idx)

    if temp_node_label ==0:
        item_ids_of_feature_vecs.append(temp_node_id)
        item_feature_for_lookup.append(temp_embed_lookup_ids)
    else:
        user_ids_of_feature_vecs.append(temp_node_id)
        user_feature_for_loopup.append(temp_embed_lookup_ids)


temp_item_input = torch.LongTensor(item_feature_for_lookup)
temp_item_vecs = embedding_lookup_table(temp_item_input)
print(temp_item_vecs.shape)
temp_x_dims = temp_item_vecs.shape[0]
temp_y_dims = temp_item_vecs.shape[1] * temp_item_vecs.shape[2]

temp_item_vecs = temp_item_vecs.reshape([temp_x_dims, temp_y_dims])
print(temp_item_vecs.shape)

temp_user_input = torch.LongTensor(user_feature_for_loopup)
temp_user_vecs = embedding_lookup_table(temp_user_input)
print(temp_user_vecs.shape)

temp_x_dims = temp_user_vecs.shape[0]
temp_y_dims = temp_user_vecs.shape[1] * temp_user_vecs.shape[2]

temp_user_vecs = temp_user_vecs.reshape([temp_x_dims, temp_y_dims])
print(temp_user_vecs.shape)

item_connected_layer = nn.Linear(temp_item_vecs.shape[1], dim_node)
user_connected_layer = nn.Linear(temp_user_vecs.shape[1], dim_node)
print(item_connected_layer)
print(user_connected_layer)

item_feature_vector = item_connected_layer(temp_item_vecs)
user_feature_vector = user_connected_layer(temp_user_vecs)

print(item_feature_vector.shape)
print(user_feature_vector.shape)

item_feature_vector = item_feature_vector.cpu().detach().numpy()
user_feature_vector = user_feature_vector.cpu().detach().numpy()
all_feature_vector = np.concatenate((item_feature_vector, user_feature_vector), axis=0)

feat_vecs_filename = dir_to_load + "/feature_vectors_v.csv"
vec_file_df = pd.DataFrame(all_feature_vector)
vec_file_df.to_csv(feat_vecs_filename)
