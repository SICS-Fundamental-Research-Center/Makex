import pandas as pd
import numpy as np
import csv

def fidelity_factual_precision(df, k):
    
    grouped = df.groupby(['graph_id'])
    def compute_fid_u_v(user_group):
        user_items = user_group['rec'].tolist()
        return sum([user_items[i] for i in range(len(user_items))])

    grouped_fid_u_v = grouped.apply(compute_fid_u_v)
    precision = (1 / len(grouped_fid_u_v)) * sum(grouped_fid_u_v )
    precision2 = (1 / 1000) * sum(grouped_fid_u_v )
    print("precision@{}:".format(k), precision, precision2)

    return precision, precision2



print("the dataset...")
file_location = "../../../dataset"
dataset_name = "ciao"
print('Enter gnn model name:')
modelname = input()

res_filename = 'sx_result/' + modelname + '/' 
dir_to_load = file_location + "/" + dataset_name + "/" 
resdir_to_load = file_location + "/" + dataset_name + "/" + res_filename + "/"

eval_filename ='/eval.csv'
eval_df = pd.read_csv(resdir_to_load + eval_filename, names= ['graph_id','pred_score', 'rec'])

max_k = 10

fid_columns = ['fid1','fid2']
fid_filename = "/fidelity_pre.csv"
with open(resdir_to_load + fid_filename, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fid_columns)
        cur_fid, cur_fid2 = fidelity_factual_precision(eval_df, max_k)
        temp_dict = {
                'fid1': cur_fid, 
                'fid2': cur_fid2
        }
        writer.writerow(temp_dict)
        print("Fidelity computation finished...")