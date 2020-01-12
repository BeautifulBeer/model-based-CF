#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math


# In[ ]:


# predict a movie rating of a user
def predict(u, i, fm_users, userMapper, fm_movies, movieMapper):
    return np.dot(fm_users[userMapper[u], :], np.transpose(fm_movies[movieMapper[i], :]))

# return NaN if count is zero
def test_error(test_data, fm_users, userMapper, fm_movies, movieMapper):
    total = 0
    count = 0
    test_data = test_data
    for idx in test_data.index:
        if test_data.iloc[idx].userId in userMapper and test_data.iloc[idx].movieId in movieMapper:
            total += pow(test_data.iloc[idx].rating - predict(test_data.iloc[idx].userId, test_data.iloc[idx].movieId, fm_users, userMapper, fm_movies, movieMapper), 2)
            count += 1
    return math.sqrt(total / count)

