import pandas as pd
import numpy as np
import os
import time


def union_topk_v_expand(input_file, output_file, topk_df_file, topk_threshold, m_threshold):
    df = pd.read_csv(input_file, low_memory=False)
    string_columns = ['title:string', 'genres:string', 'gender:string', 'zip-code:string']

    for column in string_columns:
        df[column] = df[column].fillna('').astype(str)
        df[column] = df[column].replace({np.nan: '', None: ''})
    filtered_df = df

    grouped_df = filtered_df.groupby(['pair_id', 'tie_id', 'vertex_id', 'topk']).agg({
        'pivot_x': 'first',
        'pivot_y': 'first',
        'explanation_score': 'sum',
        'label_id:int': 'first',
        'title:string': lambda x: str(x.iloc[0]),
        'genres:string': lambda x: str(x.iloc[0]),
        'avgrating:double': 'first',
        'year:double': 'first',
        'gender:string': lambda x: str(x.iloc[0]),
        'age:double': 'first',
        'occupation:int': 'first',
        'zip-code:string': lambda x: x.iloc[0]
    }).reset_index()

    grouped_df['pair_id'] = grouped_df['pair_id'].astype('Int64')
    grouped_df['pivot_x'] = grouped_df['pivot_x'].astype('Int64')
    grouped_df['pivot_y'] = grouped_df['pivot_y'].astype('Int64')
    grouped_df['topk'] = grouped_df['topk'].astype('Int64')
    grouped_df['tie_id'] = grouped_df['tie_id'].astype('Int64')
    grouped_df['explanation_score'] = grouped_df['explanation_score'].astype('Float64')

    grouped_df['vertex_id'] = grouped_df['vertex_id'].astype('Int64')
    grouped_df['label_id:int'] = grouped_df['label_id:int'].astype('Int64')
    grouped_df['avgrating:double'] = grouped_df['avgrating:double'].astype('Float64')
    grouped_df['year:double'] = grouped_df['year:double'].astype('Float64')
    grouped_df['age:double'] = grouped_df['age:double'].astype('Float64')
    grouped_df['occupation:int'] = grouped_df['occupation:int'].astype('Float64')


    def convert_to_original_format(val):
        try:
            if float(val).is_integer():
                return str(int(float(val)))
            return str(val)
        except (ValueError, TypeError):
            return str(val)



    for column in string_columns:
        grouped_df[column] = grouped_df[column].apply(convert_to_original_format)

    grouped_df.sort_values(['pair_id', 'tie_id', 'explanation_score'], ascending=[True, True, False], inplace=True)

    result_df = grouped_df[['pair_id', 'pivot_x', 'pivot_y', 'topk', 'tie_id', 'explanation_score', 'vertex_id', 'label_id:int',
                             'title:string', 'genres:string', 'avgrating:double', 'year:double',
                             'gender:string', 'age:double', 'occupation:int', 'zip-code:string']]
    result_df = result_df.sort_values(by=['pair_id', 'topk'], ascending=[True, True])
    result_df.to_csv(output_file, index=False)


    explanation_score_sum = result_df.groupby(['pair_id', 'tie_id'])['explanation_score'].sum().reset_index()
    explanation_score_sum.rename(columns={'explanation_score': 'total_explanation_score'}, inplace=True)

    merged_df = pd.merge(result_df, explanation_score_sum, on=['pair_id', 'tie_id'])
    merged_df.sort_values(by=['pair_id', 'total_explanation_score'], ascending=[True, False], inplace=True)

    merged_df = merged_df.sort_values(by=['pair_id', 'topk'], ascending=[True, True])
    merged_df.to_csv(topk_df_file, index=False)
    








def union_topk_e_expand(input_file, output_file, topk_df_file_e, topk_df_file_v, topk_threshold, m_threshold):
    df = pd.read_csv(input_file, low_memory=False)
    filtered_df = df

    grouped_df = filtered_df.groupby(['pair_id', 'tie_id', 'topk', 'source_id:int', 'target_id:int']).agg({
        'pivot_x': 'first',
        'pivot_y': 'first',
        'explanation_score': 'sum',
        'label_id:int': 'first'
    }).reset_index()

    grouped_df['pair_id'] = grouped_df['pair_id'].astype('Int64')
    grouped_df['pivot_x'] = grouped_df['pivot_x'].astype('Int64')
    grouped_df['pivot_y'] = grouped_df['pivot_y'].astype('Int64')
    grouped_df['topk'] = grouped_df['topk'].astype('Int64')
    grouped_df['tie_id'] = grouped_df['tie_id'].astype('Int64')
    grouped_df['explanation_score'] = grouped_df['explanation_score'].astype('Float64')
    grouped_df['source_id:int'] = grouped_df['source_id:int'].astype('Int64')
    grouped_df['target_id:int'] = grouped_df['target_id:int'].astype('Int64')
    grouped_df['label_id:int'] = grouped_df['label_id:int'].astype('Int64')

    result_df = grouped_df.replace({np.nan: None}).fillna('')
    result_df = grouped_df[['pair_id', 'pivot_x', 'pivot_y', 'topk', 'tie_id', 'explanation_score', 'source_id:int', 'target_id:int', 'label_id:int']]

    result_df = result_df.sort_values(by=['pair_id', 'topk'], ascending=[True, True])
    result_df.to_csv(output_file, index=False)


    df_v = pd.read_csv(topk_df_file_v, low_memory=False)

    final_df = result_df[result_df[['pair_id', 'tie_id']].apply(tuple, 1).isin(df_v[['pair_id', 'tie_id']].apply(tuple, 1))]

    final_df = final_df[['pair_id', 'pivot_x', 'pivot_y', 'topk', 'source_id:int', 'target_id:int', 'label_id:int']]
    final_df = final_df.sort_values(by=['pair_id', 'topk'], ascending=[True, True])

    final_df.to_csv(topk_df_file_e, index=False)




def union_topk_v(input_file, output_file, topk_makex_union_v_tie_id, topk_threshold):
    df = pd.read_csv(input_file, low_memory=False)
    string_columns = ['title:string', 'genres:string', 'gender:string', 'zip-code:string']

    for column in string_columns:
        df[column] = df[column].fillna('').astype(str)
        df[column] = df[column].replace({np.nan: '', None: ''})
    filtered_df = df

    grouped_df = filtered_df.groupby(['pair_id', 'vertex_id', 'topk']).agg({
        'pivot_x': 'first',
        'pivot_y': 'first',
        'tie_id': 'first',
        'label_id:int': 'first',
        'title:string': lambda x: str(x.iloc[0]),
        'genres:string': lambda x: str(x.iloc[0]),
        'avgrating:double': 'first',
        'year:double': 'first',
        'gender:string': lambda x: str(x.iloc[0]),
        'age:double': 'first',
        'occupation:int': 'first',
        'zip-code:string': lambda x: x.iloc[0]
    }).reset_index()

    grouped_df['pair_id'] = grouped_df['pair_id'].astype('Int64')
    grouped_df['pivot_x'] = grouped_df['pivot_x'].astype('Int64')
    grouped_df['pivot_y'] = grouped_df['pivot_y'].astype('Int64')
    grouped_df['topk'] = grouped_df['topk'].astype('Int64')
    grouped_df['tie_id'] = grouped_df['tie_id'].astype('Int64')
    grouped_df['vertex_id'] = grouped_df['vertex_id'].astype('Int64')
    grouped_df['label_id:int'] = grouped_df['label_id:int'].astype('Int64')

    grouped_df['avgrating:double'] = grouped_df['avgrating:double'].astype('Float64')
    grouped_df['year:double'] = grouped_df['year:double'].astype('Float64')
    grouped_df['age:double'] = grouped_df['age:double'].astype('Float64')
    grouped_df['occupation:int'] = grouped_df['occupation:int'].astype('Float64')


    def convert_to_original_format(val):
        try:
            if float(val).is_integer():
                return str(int(float(val)))
            return str(val)
        except (ValueError, TypeError):
            return str(val)



    for column in string_columns:
        grouped_df[column] = grouped_df[column].apply(convert_to_original_format)

    result_df = grouped_df[['pair_id', 'pivot_x', 'pivot_y', 'topk', 'vertex_id', 'label_id:int', 
                            'title:string', 'genres:string', 'avgrating:double', 'year:double',
                             'gender:string', 'age:double', 'occupation:int', 'zip-code:string']]

    result_df_tie_id = grouped_df[['pair_id', 'pivot_x', 'pivot_y', 'topk', 'tie_id', 'vertex_id', 'label_id:int', 
                            'title:string', 'genres:string', 'avgrating:double', 'year:double',
                             'gender:string', 'age:double', 'occupation:int', 'zip-code:string']]
    result_df = result_df.sort_values(by=['pair_id', 'topk'], ascending=[True, True])

    result_df.to_csv(output_file, index=False)

    result_df_tie_id = result_df_tie_id.sort_values(by=['pair_id', 'topk'], ascending=[True, True])
    result_df_tie_id.to_csv(topk_makex_union_v_tie_id, index=False)




def union_topk_e(input_file, output_file, topk_threshold):
    df = pd.read_csv(input_file, low_memory=False)
    filtered_df = df

    grouped_df = filtered_df.groupby(['pair_id', 'topk', 'source_id:int', 'target_id:int']).agg({
        'pivot_x': 'first',
        'pivot_y': 'first',
        'label_id:int': 'first'
    }).reset_index()

    grouped_df['pair_id'] = grouped_df['pair_id'].astype('Int64')
    grouped_df['pivot_x'] = grouped_df['pivot_x'].astype('Int64')
    grouped_df['pivot_y'] = grouped_df['pivot_y'].astype('Int64')
    grouped_df['topk'] = grouped_df['topk'].astype('Int64')
    grouped_df['source_id:int'] = grouped_df['source_id:int'].astype('Int64')
    grouped_df['target_id:int'] = grouped_df['target_id:int'].astype('Int64')
    grouped_df['label_id:int'] = grouped_df['label_id:int'].astype('Int64')

    result_df = grouped_df.replace({np.nan: None}).fillna('')
    result_df = grouped_df[['pair_id', 'pivot_x', 'pivot_y', 'topk', 'source_id:int', 'target_id:int', 'label_id:int']]


    result_df = result_df.sort_values(by=['pair_id', 'topk'], ascending=[True, True])
    result_df.to_csv(output_file, index=False)




def average_element_difference_between_lists(lists):
    num_lists = len(lists)
    if num_lists < 2:
        return 0

    total_difference = 0
    pair_count = 0

    for i in range(num_lists):
        for j in range(i+1, num_lists):
            set_i = set(lists[i])
            set_j = set(lists[j])
            symmetric_diff = set_i.symmetric_difference(set_j)
            difference = len(symmetric_diff) / max(len(set_i), len(set_j))
            total_difference += difference
            pair_count += 1

    average_difference = total_difference / pair_count
    return average_difference




def change_v_attributes(makex_explanation_v, makex_explanation_v_new, original_v_file):
    file1 = pd.read_csv(makex_explanation_v)
    file2 = pd.read_csv(original_v_file)
    merged_df = file1.merge(file2, on='vertex_id', suffixes=('', '_file2'))
    columns_to_replace = ['title:string', 'genres:string', 'avgrating:double', 'year:double',
                        'gender:string', 'age:double', 'occupation:int', 'zip-code:string']
    for col in columns_to_replace:
        condition = merged_df[col].notna()
        merged_df.loc[condition, col] = merged_df.loc[condition, col+'_file2']

    merged_df.drop([col+'_file2' for col in columns_to_replace], axis=1, inplace=True)
    merged_df.drop(['label_id'], axis=1, inplace=True)
    merged_df.sort_values(by='pair_id', inplace=True)

    merged_df.to_csv(makex_explanation_v_new, index=False)


if __name__ == "__main__":

    model = "hgt"
    explanation_dir="/"+ model +"/local_explanation/"
    topk_list = [1, 5, 10, 15]

    union_folder_list = []
    for topk in topk_list:
        path = explanation_dir + "topk" +str(topk) + "/"
        union_folder_list.append(path)

    k_ = 0
    for union_folder in union_folder_list:

        topk = topk_list[k_]
        k_ += 1
        topm_list = [1]
        for topm in topm_list:
            topk_threshold = topk
            m_threshold = topm

            t_begin = time.time()
            
            original_v_file = "./v_original.csv"


            makex_explanation_v = union_folder + "v.csv"
            makex_explanation_v_new = union_folder + "v_new.csv"

            makex_explanation_e = union_folder + "e.csv"
            makex_explanation_v_union = union_folder + "v_union_topm" + str(topm) + ".csv"
            makex_explanation_e_union = union_folder + "e_union_topm" + str(topm) + ".csv"
            topk_df_file_v = union_folder + "v_union_topk"+str(topk)+"_topm" + str(topm) + ".csv"


            topk_makex_union_v = union_folder + "topk"+str(topk)+"_topm" + str(topm)+ "/v.csv"
            directory_path = os.path.dirname(topk_makex_union_v)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            topk_makex_union_v_tie_id = union_folder + "v_union_topk"+str(topk)+"_topm" + str(topm) + "_tie_id.csv"
            

            topk_makex_union_e = union_folder + "topk"+str(topk)+"_topm" + str(topm)+ "/e.csv"
            directory_path = os.path.dirname(topk_makex_union_e)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            change_v_attributes(makex_explanation_v, makex_explanation_v_new, original_v_file)
            makex_explanation_v = makex_explanation_v_new
            

            union_topk_v_expand(makex_explanation_v, makex_explanation_v_union, topk_df_file_v, topk_threshold, m_threshold)
            union_topk_v(topk_df_file_v, topk_makex_union_v,topk_makex_union_v_tie_id,topk_threshold)

            union_topk_e_expand(makex_explanation_e, makex_explanation_e_union, topk_makex_union_e, topk_makex_union_v_tie_id, topk_threshold, m_threshold)

            t_end = time.time()
            print("topk {} all pair local explanation time: {}".format(topk, (t_end - t_begin)))
                                                                                                                                                                                                                                                              