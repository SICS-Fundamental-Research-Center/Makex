
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
    
    return temp_dict

def change_value_movielens(dir_to_load, attributes):

    avgrate = attributes['avgrating:double'].to_numpy()
    avgrate = np.unique(avgrate)
    new_rate = np.round(avgrate, 1)

    attributes['avgrating:double'] = attributes['avgrating:double'].replace(avgrate, new_rate)

    new_attr_filename = dir_to_load + "/new_v.csv"
    attributes.to_csv(new_attr_filename, index=False)


def change_value_yelp(dir_to_load, attributes):
    num_items  = 45538
    num_users =  91457

    latitude = attributes['latitude:double'].to_numpy()
    latitude = np.unique(latitude)
    new_lat = np.round(latitude, 0)

    attributes['latitude:double'] = attributes['latitude:double'].replace(latitude, new_lat)

    longitude = attributes['longitude:double'].to_numpy()
    longitude = np.unique(longitude)
    new_long = np.round(longitude, 0)

    attributes['longitude:double'] = attributes['longitude:double'].replace(longitude, new_long)


    new_attr_filename = dir_to_load + "/new_v.csv"
    attributes.to_csv(new_attr_filename, index=False)

def save_new_values(dir_to_load, dataset):
    attr_filename =  "/" + "v.csv"

    attributes = pd.read_csv(dir_to_load + attr_filename)
    columns_to_convert = ['pair_id', 'pivot_x', 'pivot_y', 'topk', 'vertex_id', 'label_id:int']
    attributes[columns_to_convert] = attributes[columns_to_convert].astype(int)
    if dataset == 'movielens':
        change_value_movielens(dir_to_load, attributes)
    elif dataset == 'yelp':
        change_value_yelp(dir_to_load, attributes)
    else:
        print("data set not found")


print("the dataset...")
file_location = "../../../dataset"
dataset_name = 'movielens'
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

    attr_filename = "/new_v.csv"

    dir_to_load_attr = file_location + "/" + dataset_name
    attr_map_filename = "/attribute_to_embedding_mapping.csv"

    if os.path.isfile(dir_to_load + attr_filename) == False:
        save_new_values(dir_to_load, dataset_name)

    data_types = {
        'pair_id': 'Int64',
        'pivot_x': 'Int64',
        'pivot_y': 'Int64',
        'topk': 'Int64',
        'vertex_id': 'Int64',
        'label_id:int': 'Int64',
        'title:string': str,
        'genres:string': str,
        'avgrating:double': 'Float64',
        'year:double': 'Float64',
        'gender:string': str,
        'age:double': 'Float64',
        'occupation:int': 'Float64',
        'zip-code:string': str
    }

    attributes = pd.read_csv(dir_to_load + attr_filename, dtype=data_types)

    attr_mapping =  pd.read_csv(dir_to_load_attr + attr_map_filename, header=None).to_numpy()

    cnt_values_in_attr = attr_mapping.size

    dim_attr = 16
    dim_node = 16

    fields_name = ['title', 'genres', 'avgrating', 'year', 'gender', 'age', 'occupation', 'zip-code']
    titles = attributes['title:string'].to_numpy()
    genres = attributes['genres:string'].to_numpy()
    avgratings = attributes['avgrating:double'].to_numpy()
    years = attributes['year:double'].to_numpy()
    genders = attributes['gender:string'].to_numpy()
    ages = attributes['age:double'].to_numpy()
    occupation = attributes['occupation:int'].to_numpy()
    zipcodes = attributes['zip-code:string'].to_numpy(dtype=str)

    attributes = np.stack((titles, genres, avgratings, years, genders, ages, occupation, zipcodes), axis=1)
    print(attributes.shape)

    attr_mapping_dict = convert_numpy_to_dict(attr_mapping)
    embedding_lookup_table = nn.Embedding(cnt_values_in_attr + 8, dim_attr) 

    ids_of_feature_vecs = [] 
    feature_for_lookup = []  

    for i in range(attributes.shape[0]):

        temp_embed_lookup_ids = []
        temp_pair = attributes[i]
        for j in range(attributes.shape[1]):
            temp_field = fields_name[j]
            temp_value = temp_pair[j]
            if temp_field == 'title':
                temp_embed_lookup_ids.append(cnt_values_in_attr)
            elif pd.isna(temp_value) or temp_value == 'nan':
                temp_embed_lookup_ids.append(cnt_values_in_attr + j) 
            else:
                if temp_field == 'zip-code' and len(temp_value) < 10:
                    temp_value = int(float(temp_value))
                    if temp_value < 10000:
                        temp_value = str('0' + str(temp_value))
                mapped_idx = attr_mapping_dict[(temp_field, str(temp_value))]
                temp_embed_lookup_ids.append(mapped_idx)
            
        ids_of_feature_vecs.append(i)
        feature_for_lookup.append(temp_embed_lookup_ids)


    temp_input = torch.LongTensor(feature_for_lookup)
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