#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os


# In[2]:


# root mean squared error
def RMSE(user_item, users, movies, regular_term):
    total = 0
    count = 0
    for user in user_item:
        for movie, score in user_item[user]:
            movie = int(movie)
            user = int(user)
            square = pow(score - np.dot(users[user, :], np.transpose(movies[movie, :])), 2)
            regular = np.sum(regular_term *(np.power(users[user, :], 2) + np.power(movies[movie, :], 2)))
            total = total + square + regular
            count = count + 1
    return math.sqrt(total / count)

def load_files(user_count, movie_count, latent_dim):
    files = []
    if os.path.isfile('./model/latent_user.npy'):
        fm_users = np.load('./model/latent_user.npy')
        files.append(fm_users)
    else:
        files.append(np.random.rand(user_count, latent_dim))
    if os.path.isfile('./model/latent_movie.npy'):
        fm_movies = np.load('./model/latent_movie.npy')
        files.append(fm_movies)
    else:
        files.append(np.random.rand(movie_count, latent_dim))
    if os.path.isfile('./model/train_error.npy'):
        train_error = np.load('./model/train_error.npy')    
        files.append(train_error)
    else:
        files.append([])
    return files

def save_files(fm_users, fm_movies, train_error):
    np.save('./model/latent_user.npy', fm_users)
    np.save('./model/latent_movie.npy', fm_movies)
    np.save('./model/train_error.npy', train_error)
#     with open('./model/userMapper.json', 'w') as f:
#         json.dump(userMapper, f)
#     with open('./model/movieMapper.json', 'w') as f:
#         json.dump(movieMapper, f)
#     user_item_json = {}
#     for key in user_item:
#         user_item_json[key] = user_item[key].tolist()
#     with open('./model/user_item.json', 'w') as f:
#         json.dump(user_item_json, f)


# In[ ]:




