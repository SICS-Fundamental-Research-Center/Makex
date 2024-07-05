
import os
import csv

os.environ["DGLBACKEND"] = "pytorch"
import pandas as pd
import numpy as np

def change_value_movielens(dir_to_load, attributes):
    avgrate = attributes[:,4]
    print(avgrate.size)
    for idx in range(avgrate.size):
        value = attributes[idx, 4]
        if pd.isna(value):
            break
        attributes[idx, 4] = round(value, 1)

    new_attr_filename = dir_to_load + "/new_attributes_v.csv"
    df = pd.DataFrame(attributes, columns=['vertex_id', 'label_id', 'title', 'genres', 'avgrating', 'year', 'gender', 'age', 'occupation', 'zip-code'])
    df.to_csv(new_attr_filename, index=False)

def change_value_yelp(dir_to_load, attributes):
    num_items  = 45538
    num_users =  91457

    for idx in range(num_items):
        temp_pc = attributes[idx, 2]
        if pd.isna(temp_pc):
            attributes[idx, 2] = 'empty'

        temp_latitude = attributes[idx, 3]
        temp_longtitude = attributes[idx, 4]

        if pd.isna(temp_latitude):
            temp_latitude = 0
        
        if pd.isna(temp_longtitude):
            temp_longtitude = 0

        attributes[idx, 3] = round(temp_latitude, 0)
        attributes[idx, 4] = round(temp_longtitude, 0)


    new_attr_filename = dir_to_load + "/new_attributes_v.csv"
    df = pd.DataFrame(attributes, columns=['vertex_id', 'label_id', 'postal_code', 'latitude', 'longitude', 'item_review_count', 'is_open', 'user_review_count', 'yelping_since', 'fans', 'average_stars', 'name'])
    df.to_csv(new_attr_filename, index=False)

print("the dataset...")
file_location = "../../dataset"
dataset_name = "movielens"
dir_to_load = file_location + "/" + dataset_name
attr_filename =  "/" + dataset_name + "_v.csv"

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
attributes = pd.read_csv(dir_to_load + attr_filename, dtype=dtypes, low_memory=False).to_numpy()

if dataset_name == "movielens":
    change_value_movielens(dir_to_load, attributes)
elif dataset_name == "yelp":
    print("start")
    change_value_yelp(dir_to_load, attributes)
    print("finish")
else:
    print("dataset not found")
