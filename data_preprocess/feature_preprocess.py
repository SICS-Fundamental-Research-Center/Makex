import pandas as pd


#Feature preprocess for yelp_v.csv
csv_file = "../DataSets/Movielens/original_graph/yelp_v.csv"
df = pd.read_csv(csv_file)

def process_postal_code(code):
    if pd.isna(code):
        return None
    if code.isnumeric():
        return code[:2]
    else:
        return code[0]


df.loc[df['label_id:int'] == 0, 'wllabel'] = df.loc[df['label_id:int'] == 0, 'wllabel'].apply(process_postal_code)


postal_code_split = ['85','89', 'M', '28', '44']
df.loc[df['label_id:int'] == 0, 'wllabel'] = df.loc[df['label_id:int'] == 0, 'wllabel'].apply(lambda x: '00' if x not in postal_code_split else x)

column_names = ['wllabel']
for column in df.columns:
    if column in column_names:
        value_counts = df[column].value_counts()
        print(value_counts)
        print()



df.loc[df['label_id:int'] == 0, 'latitude:double'] = df.loc[df['label_id:int'] == 0, 'latitude:double'].astype(str)
def process_postal_code(code):
    if pd.isna(code):
        return None
    if code.isnumeric():
        return code[:2]
    else:
        return code[0]


df.loc[df['label_id:int'] == 0, 'latitude:double'] = df.loc[df['label_id:int'] == 0, 'latitude:double'].apply(process_postal_code)

column_names = ['latitude:double']
for column in df.columns:
    if column in column_names:

        value_counts = df[column].value_counts()
        print(value_counts)
        print()


df.loc[df['label_id:int'] == 0, 'longitude:double'] = df.loc[df['label_id:int'] == 0, 'longitude:double'].astype(str)

def extract_first_two_chars(value):
    if pd.isna(value):
        return None
    if value.startswith('-'):
        return value[:2]
    else:
        return value[:1]

df.loc[df['label_id:int'] == 0, 'longitude:double'] = df.loc[df['label_id:int'] == 0, 'longitude:double'].apply(extract_first_two_chars)


column_names = ['longitude:double']
for column in df.columns:
    if column in column_names:

        value_counts = df[column].value_counts()
        print(value_counts)
        print()

item_review_count_split= [20, 50, 80]


for i in range(len(item_review_count_split)):
    if i == 0 :
        filtered_df = df[(df['item_review_count:int'] < item_review_count_split[i])]
        count_rows = filtered_df.shape[0]
    if 0 < i < 3:
        filtered_df = df[(df['item_review_count:int'] >= item_review_count_split[i-1]) & (df['item_review_count:int'] <= item_review_count_split[i])]
        
        count_rows = filtered_df.shape[0]



for i in range(len(item_review_count_split)):
    if i == 0:
        df.loc[df['item_review_count:int'] <= item_review_count_split[i], 'item_review_count:int'] = int(item_review_count_split[i])
    else:
        df.loc[(df['item_review_count:int'] > item_review_count_split[i-1]) & (df['item_review_count:int'] <= item_review_count_split[i]), 'item_review_count:int'] = int(item_review_count_split[i])
df.loc[df['item_review_count:int'] > 80, 'item_review_count:int'] = 90

column_names = ['item_review_count:int']
for column in df.columns:
    if column in column_names:

        value_counts = df[column].value_counts()
        print(value_counts)
        print()

column_names = ['is_open:int']
for column in df.columns:
    if column in column_names:

        value_counts = df[column].value_counts()
        print(value_counts)
        print()

user_review_count_split= [20, 50, 80]


for i in range(len(user_review_count_split)):
    if i == 0 :
        filtered_df = df[(df['user_review_count:int'] < user_review_count_split[i])]
        count_rows = filtered_df.shape[0]

    if 0 < i < 3:
        filtered_df = df[(df['user_review_count:int'] >= user_review_count_split[i-1]) & (df['user_review_count:int'] <= user_review_count_split[i])]
        
        count_rows = filtered_df.shape[0]

max_num = 90
filtered_df = df[(df['user_review_count:int'] > max_num)]
count_rows = filtered_df.shape[0]

min_num = 20
filtered_df = df[(df['user_review_count:int'] < min_num)]
count_rows = filtered_df.shape[0]






for i in range(len(user_review_count_split)):
    if i == 0:
        df.loc[df['user_review_count:int'] <= user_review_count_split[i], 'user_review_count:int'] = user_review_count_split[i]
    else:
        df.loc[(df['user_review_count:int'] > user_review_count_split[i-1]) & (df['user_review_count:int'] <= user_review_count_split[i]), 'user_review_count:int'] = user_review_count_split[i]
df.loc[df['user_review_count:int'] > 80, 'user_review_count:int'] = 90

column_names = ['user_review_count:int']
for column in df.columns:
    if column in column_names:

        value_counts = df[column].value_counts()
        print(value_counts)
        print()

df['yelping_since:string'] = df['yelping_since:string'].astype(str)

def process_postal_code(code):
    if pd.isna(code):
        return None
    if code.isnumeric():
        return code[:4]
    else:
        return code[:4]

df['yelping_since:string'] = df['yelping_since:string'].apply(process_postal_code)
df['yelping_since:string'] = df['yelping_since:string'].astype(float)

#user_review_count
yelping_since_split= [2011, 2013, 2018]

for i in range(len(yelping_since_split)):
    if i == 0 :
        filtered_df = df[(df['yelping_since:string'] <= yelping_since_split[i])]
        count_rows = filtered_df.shape[0]
    if 0 < i < 3:
        filtered_df = df[(df['yelping_since:string'] > yelping_since_split[i-1]) & (df['yelping_since:string'] <= yelping_since_split[i])]
        
        count_rows = filtered_df.shape[0]



max_num = 2018
filtered_df = df[(df['yelping_since:string'] > max_num)]
count_rows = filtered_df.shape[0]

min_num = 2000
filtered_df = df[(df['yelping_since:string'] < min_num)]
count_rows = filtered_df.shape[0]



for i in range(len(yelping_since_split)):
    if i == 0:
        df.loc[df['yelping_since:string'] <= yelping_since_split[i], 'yelping_since:string'] = str(yelping_since_split[i])
    else:
        df.loc[(df['yelping_since:string'] > yelping_since_split[i-1]) & (df['yelping_since:string'] <= yelping_since_split[i]), 'yelping_since:string'] = str(yelping_since_split[i])

column_names = ['yelping_since:string']
for column in df.columns:
    if column in column_names:

        value_counts = df[column].value_counts()
        print(value_counts)
        print()


#fans
fans_split= [0, 2]


for i in range(len(fans_split)):
    if i == 0 :
        filtered_df = df[(df['fans:int'] <= fans_split[i])]
        count_rows = filtered_df.shape[0]
    else:
        filtered_df = df[(df['fans:int'] > fans_split[i-1]) & (df['fans:int'] <= fans_split[i])]
    
        count_rows = filtered_df.shape[0]
max_num = 2
filtered_df = df[(df['fans:int'] > max_num)]
count_rows = filtered_df.shape[0]


min_num = 0
filtered_df = df[(df['fans:int'] <= min_num)]
count_rows = filtered_df.shape[0]


for i in range(len(fans_split)):
    if i == 0:
        df.loc[df['fans:int'] <= fans_split[i], 'fans:int'] = fans_split[i]
    else:
        df.loc[(df['fans:int'] > fans_split[i-1]) & (df['fans:int'] <= fans_split[i]), 'fans:int'] = fans_split[i]
df.loc[df['fans:int'] > 2, 'fans:int'] = 5

column_names = ['fans:int']
for column in df.columns:
    if column in column_names:

        value_counts = df[column].value_counts()
        print(value_counts)
        print()

average_stars_split= [3.5, 4]


for i in range(len(average_stars_split)):
    if i == 0 :
        filtered_df = df[(df['average_stars:double'] <= average_stars_split[i])]
        count_rows = filtered_df.shape[0]
    else:
        filtered_df = df[(df['average_stars:double'] > average_stars_split[i-1]) & (df['average_stars:double'] <= average_stars_split[i])]
        count_rows = filtered_df.shape[0]
max_num = 4
filtered_df = df[(df['average_stars:double'] > max_num)]
count_rows = filtered_df.shape[0]


min_num = 3.5
filtered_df = df[(df['average_stars:double'] <= min_num)]
count_rows = filtered_df.shape[0]


for i in range(len(average_stars_split)):
    if i == 0:
        df.loc[df['average_stars:double'] <= average_stars_split[i], 'average_stars:double'] = average_stars_split[i]
    else:
        df.loc[(df['average_stars:double'] > average_stars_split[i-1]) & (df['average_stars:double'] <= average_stars_split[i]), 'average_stars:double'] = average_stars_split[i]
df.loc[df['average_stars:double'] > 4, 'average_stars:double'] = 5

column_names=['average_stars:double']
for column in df.columns:
    if column in column_names:

        value_counts = df[column].value_counts()
        print(value_counts)
        print()



output_file = "../DataSets/Movielens/original_graph/yelp_v_modified.csv"
df.to_csv(output_file, index=False)
