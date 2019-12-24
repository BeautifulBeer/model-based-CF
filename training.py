#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numba import njit, prange
from tools import RMSE


# In[2]:


# Stochastic Gradient using numba
@njit
def stochastic_gradient(user_item, users, user_count, movies, learning_rate, regular_term):
    for user in prange(user_count):
        movie_length = user_item[user].shape[0]
        for idx in prange(movie_length):
            movie = int(user_item[user][idx][0])
            score = user_item[user][idx][1]            
            error = score - np.dot(movies[movie, :], np.transpose(users[user, :]))
            delta_movie = users[user, :] * error - movies[movie, :] * regular_term
            delta_user = movies[movie, :] * error - users[user, :] * regular_term
            movies[movie, :] = movies[movie, :] + delta_movie * learning_rate
            users[user, :] = users[user, :] + delta_user * learning_rate

# 1 epoch training, return lossValue
def training(user_item, fm_users, user_count, fm_movies, learning_rate, regular_term):
    stochastic_gradient(user_item, fm_users, user_count, fm_movies, learning_rate, regular_term)
    return RMSE(user_item, fm_users, fm_movies, regular_term)


# In[ ]:




