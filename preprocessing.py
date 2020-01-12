#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import os
import json
from numba import types
from numba.typed import Dict


# In[1]:


# load origin data and shuffle
def shuffle_data():
    ratings = pd.DataFrame(pd.read_csv('./ml-latest/ratings.csv'))
    ratings = ratings.sample(frac=1).reset_index(drop=True)
    ratings.to_csv('shuffled_ratings.csv', index=False)

# user-item matrix
def build_adjacency_matrix(df, userMapper, movieMapper):
    user_item = Dict.empty(
            key_type=types.int64,
            value_type=types.float64[:, :]
    )
    for row in df.index:
        userId = userMapper[int(df.iloc[row].userId)]
        movieId = movieMapper[int(df.iloc[row].movieId)]
        rating = df.iloc[row].rating
        if userId not in user_item:
            user_item[userId] = np.array([[movieId, rating]], dtype=np.float64)
        else:
            user_item[userId] = np.concatenate((user_item[userId], np.array([[movieId, rating]], dtype=np.float64)), axis=0)
    return user_item

# movieId convert to new index
def movieIdMapper(ratings):
    if os.path.isfile('./model/movieMapper.json'):
        mapper = open('./model/movieMapper.json').read()
        return json.loads(mapper)
    else:
        movieIdSet = ratings.movieId.unique()
        movieIdSet.sort()
        mapper = {}
        idx = 0
        for id in movieIdSet:
            mapper[int(id)] = idx
            idx = idx + 1
        return mapper

# userId convert to new index
def userIdMapper(ratings):
    if os.path.isfile('./model/userMapper.json'):
        mapper = open('./model/userMapper.json').read()
        return json.loads(mapper)
    else:
        userIdSet = ratings.userId.unique()
        userIdSet.sort()
        mapper = {}
        idx = 0
        for id in userIdSet:
            mapper[int(id)] = idx
            idx = idx + 1
        return mapper

