import pandas as pd
import json
import numpy as np
import ast
import hunspell
hspell = hunspell.Hunspell('/home/thsamara/anaconda3/envs/aspect_s/lib/python3.11/site-packages/hunspell/dictionaries/el_GR')
columns_to_check  =  ['Ανάλυση οθόνης',
 'Καταγραφή Video',
 'Μουσική',
 'Μπαταρία',
 'Ποιότητα κλήσης',
 'Σχέση ποιότητας τιμής',
 'Ταχύτητα',
 'Φωτογραφίες']
def take_aspect_list(df):

    all_pros = set()
    all_soso = set()
    all_cons = set()

    for val in df['agg_rating']:
        if isinstance(val, str) and val.startswith("{"):
            try:
                rating_dict = ast.literal_eval(val)
                all_pros.update(rating_dict.get('pros', []))
                all_soso.update(rating_dict.get('so-so', []))
                all_cons.update(rating_dict.get('cons', []))
            except (ValueError, SyntaxError) as e:
                print(f"Skipping invalid entry:\n{val}\nError: {e}")
    if set(all_cons) == set(all_pros) == set(all_soso):
        print(list(all_cons))
        print(list(all_pros))
        print(list(all_soso))
        return list(all_cons)
    else:
        print('Error: all_cons != all_pros != all_soso')
        print(list(all_cons))
        print(list(all_pros))
        print(list(all_soso))
        return None
    
def fill_aspect_list(df):
    all_tags = set()
    for entry in df['agg_rating']:
        parsed = ast.literal_eval(entry)
        all_tags.update(parsed['pros'], parsed['so-so'], parsed['cons'])

    for tag in all_tags:
        df[tag] = 0  # initialize with 0 or NaN depending on your use case

    # Update the values safely
    for i, entry in enumerate(df['agg_rating']):
        temp = ast.literal_eval(entry)
        for j in temp['pros']:
            df.loc[i, j] = 1
        for j in temp['so-so']:
            df.loc[i, j] = 0
        for j in temp['cons']:
            df.loc[i, j] = -1
    return df
def stars(data):

    temp = []
    temp = data["stars"].values.tolist()
    for i in range(0, len(data["stars"])):
        if int(temp[i]) <= 2:
            temp[i] = -1
        elif int(temp[i]) == 3:
            temp[i] = 0
        else:
            temp[i] = 1
    data["stars"] = temp
    return data

def clean_accent(text):

    t = text

    # el
    t = t.replace("Ά", "Α")
    t = t.replace("Έ", "Ε")
    t = t.replace("Ί", "Ι")
    t = t.replace("Ή", "Η")
    t = t.replace("Ύ", "Υ")
    t = t.replace("Ό", "Ο")
    t = t.replace("Ώ", "Ω")
    t = t.replace("ά", "α")
    t = t.replace("έ", "ε")
    t = t.replace("ί", "ι")
    t = t.replace("ή", "η")
    t = t.replace("ύ", "υ")
    t = t.replace("ό", "ο")
    t = t.replace("ώ", "ω")
    t = t.replace("ς", "σ")
    t = t.replace("♡", "")
    t = t.replace("☆", "")
    t = t.replace("*", "")

    return t


def main():
    df = pd.read_csv('rawreviews.csv')
    print((df))
    df=df.dropna(subset=['agg_rating','comment'])
    print(df)
    
    aspect_list = take_aspect_list(df)
    if aspect_list is None:
        print('Error: aspect_list is None')
        return
    for column in aspect_list:
        df[column] = '-' # fill columns of  aspects with '-'
    print(df)
    df = fill_aspect_list(df)
    print(df)
    df=df.drop('GPS',axis=1)
    #df=df.drop('topic',axis=1)
    #df=df.drop('title',axis=1)
    #df=df.drop('date',axis=1)
    #df=df.drop('verified_purchase',axis=1)
    #df=df.drop('agg_rating',axis=1)
    #df=stars(df)
    print((df))
    df_new = df[~df[columns_to_check].isin(['-']).any(axis=1)]
    for col in columns_to_check:
        print(df[col].value_counts())
        print(df_new[col].value_counts())

if __name__ == "__main__":
    main()