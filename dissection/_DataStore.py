import numpy as np
import random

from tf2_examples import dataloader
raw_data_master = dataloader.load_citeulike('C:/Users/jstep/PycharmProjects/openrec/sample_data/')

# class _DataStore(object):
# 
#     def __init__(self, raw_data, total_users, total_items, implicit_negative=True,
#                  num_negatives=None, seed=None, sortby=None, asc=True, name=None):

raw_data = raw_data_master['train_data']
total_users = raw_data_master['total_users']
total_items = raw_data_master['total_items']

name = None
random.seed(None)
sortby = None
implicit_negative = True
num_negatives = None
asc=True

if type(raw_data) == np.ndarray:
    _raw_data = raw_data
else:
    raise TypeError("Unsupported data input schema. Please use structured numpy array.")
_rand_ids = []

_total_users = total_users
_total_items = total_items

_sortby = sortby

_index_store = dict()
_implicit_negative = implicit_negative
_num_negatives = num_negatives

if _implicit_negative:
    _index_store['positive'] = dict()
    for ind, entry in enumerate(_raw_data):
        if entry['user_id'] not in _index_store['positive']:
            _index_store['positive'][entry['user_id']] = dict()
        _index_store['positive'][entry['user_id']][entry['item_id']] = ind
    _index_store['positive_sets'] = dict()
    for user_id in _index_store['positive']:
        _index_store['positive_sets'][user_id] = set(_index_store['positive'][user_id])
    if num_negatives is not None:
        _index_store['negative'] = dict()
        for user_id in _index_store['positive']:
            _index_store['negative'][user_id] = dict()
            shuffled_items = np.random.permutation(_total_items)
            for item in shuffled_items:
                if item not in _index_store['positive'][user_id]:
                    _index_store['negative'][user_id][item] = None
                if len(_index_store['negative'][user_id]) == num_negatives:
                    break
        _index_store['negative_sets'] = dict()
        for user_id in _index_store['negative']:
            _index_store['negative_sets'][user_id] = set(_index_store['negative'][user_id])
else:
    _index_store['positive'] = dict()
    _index_store['negative'] = dict()
    for ind, entry in enumerate(_raw_data):
        if entry['label'] > 0:
            if entry['user_id'] not in _index_store['positive']:
                _index_store['positive'][entry['user_id']] = dict()
            _index_store['positive'][entry['user_id']][entry['item_id']] = ind
        else:
            if entry['user_id'] not in _index_store['negative']:
                _index_store['negative'][entry['user_id']] = dict()
            _index_store['negative'][entry['user_id']][entry['item_id']] = ind
    _index_store['positive_sets'] = dict()
    for user_id in _index_store['positive']:
        _index_store['positive_sets'][user_id] = set(_index_store['positive'][user_id])
    _index_store['negative_sets'] = dict()
    for user_id in _index_store['negative']:
        _index_store['negative_sets'][user_id] = set(_index_store['negative'][user_id])

if _sortby is not None:
    _index_store['positive_sorts'] = dict()
    for user_id in _index_store['positive_sets']:
        _index_store['positive_sorts'][user_id] = sorted(list(_index_store['positive_sets'][user_id]),
                                                              key=lambda item: \
                                                                  _raw_data[
                                                                      _index_store['positive'][user_id][
                                                                          item]][_sortby],
                                                              reverse=not asc)

    # def contain_negatives(self):
    # 
    #     if _implicit_negative and _num_negatives is None:
    #         return False
    #     else:
    #         return True
    # 
    # def next_random_record(self):
    # 
    #     if len(_rand_ids) == 0:
    #         _rand_ids = list(range(len(_raw_data)))
    #         random.shuffle(_rand_ids)
    #     return _raw_data[_rand_ids.pop()]
    # 
    # def is_positive(self, user_id, item_id):
    # 
    #     if user_id in _index_store['positive'] and item_id in _index_store['positive'][user_id]:
    #         return True
    #     return False
    # 
    # def sample_positive_items(self, user_id, num_samples=1):
    # 
    #     if user_id in _index_store['positive_sets']:
    #         return random.sample(_index_store['positive_sets'][user_id], num_samples)
    #     else:
    #         return []
    # 
    def sample_negative_items(self, user_id, num_samples=1):

        if 'negative_sets' in _index_store:
            if user_id in _index_store['negative_sets']:
                return random.sample(_index_store['negative_sets'][user_id], num_samples)
            else:
                return []
        else:
            sample_id = random.randint(0, _total_items - 1)
            sample_set = set()
            while len(sample_set) < num_samples:
                if user_id not in _index_store['positive_sets'] or sample_id not in \
                        _index_store['positive_sets'][user_id]:
                    sample_set.add(sample_id)
                sample_id = random.randint(0, _total_items - 1)
            return list(sample_set)
    # 
    # def get_positive_items(self, user_id, sort=False):
    # 
    #     if user_id in _index_store['positive_sets']:
    #         if sort:
    #             assert _sortby is not None, "sortby key is not specified."
    #             return _index_store['positive_sorts'][user_id]
    #         else:
    #             return list(_index_store['positive_sets'][user_id])
    #     else:
    #         return []
    # 
    # def get_negative_items(self, user_id):
    # 
    #     if 'negative_sets' in _index_store:
    #         if user_id in _index_store['negative_sets']:
    #             return list(_index_store['negative_sets'][user_id])
    #         else:
    #             return []
    #     else:
    #         negative_items = []
    #         for item_id in range(_total_items):
    #             if item_id not in _index_store['positive_sets'][user_id]:
    #                 negative_items.append(item_id)
    #         return negative_items
    # 
    # def warm_users(self, threshold=1):
    # 
    #     users_list = []
    #     for user_id in _index_store['positive']:
    #         if len(_index_store['positive'][user_id]) >= threshold:
    #             users_list.append(user_id)
    #     return users_list
    # 
    # def total_users(self):
    # 
    #     return _total_users
    # 
    # def total_items(self):
    # 
    #     return _total_items
    # 
    # def total_records(self):
    # 
    #     return len(_raw_data)

