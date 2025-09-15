import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset


random.seed(0)

class UserRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, item_rating_tensor,bundle_tensor,bundle_rating_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.item_rating_tensor = item_rating_tensor
        self.bundle_tensor = bundle_tensor
        self.bundle_rating_tensor = bundle_rating_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.item_rating_tensor[index],self.bundle_tensor[index], self.bundle_rating_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


def align_length(user_item, user_item_rating, user_bundle):
    assert len(user_item) == len(user_item_rating), "user_item 和 rating 长度必须一致"

    target_len = len(user_bundle)
    current_len = len(user_item)

    if current_len == target_len:
        return user_item, user_item_rating

    elif current_len < target_len:
        # 需要扩展
        times = target_len // current_len
        remainder = target_len % current_len

        # 重复填充再补余数
        user_item_extended = user_item * times + user_item[:remainder]
        user_item_rating_extended = user_item_rating * times + user_item_rating[:remainder]

        return user_item_extended, user_item_rating_extended

    else:
        # 截断
        user_item_trimmed = user_item[:target_len]
        user_item_rating_trimmed = user_item_rating[:target_len]

        return user_item_trimmed, user_item_rating_trimmed

class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, item_ratings, bundle_train_ratings,bundle_test_ratings,bundle_tune_ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in item_ratings.columns
        assert 'itemId' in item_ratings.columns
        assert 'rating' in item_ratings.columns

        assert 'userId' in bundle_train_ratings.columns
        assert 'bundleId' in bundle_train_ratings.columns
        assert 'rating' in bundle_train_ratings.columns

        assert 'userId' in bundle_test_ratings.columns
        assert 'bundleId' in bundle_test_ratings.columns
        assert 'rating' in bundle_test_ratings.columns

        assert 'userId' in bundle_tune_ratings.columns
        assert 'bundleId' in bundle_tune_ratings.columns
        assert 'rating' in bundle_tune_ratings.columns


        self.item_ratings = item_ratings
        self.bundle_train_ratings = bundle_train_ratings
        self.bundle_test_ratings = bundle_test_ratings
        self.bundle_tune_ratings = bundle_tune_ratings
        # explicit feedback using _normalize and implicit using _binarize
        # self.preprocess_ratings = self._normalize(ratings)
        #self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.bundle_train_ratings['userId'].unique())
        self.item_pool = set(self.item_ratings['itemId'].unique())
        self.bundle_pool = set(self.bundle_train_ratings['bundleId'].unique())
        print(max(self.user_pool), max(self.item_pool), max(self.bundle_pool))
        if 6986 in self.user_pool:print('yes')
        # create negative item samples for NCF learning
        # 99 negatives for each user's test item
        self.item_negatives = self._sample_negative_item(item_ratings)
        self.bundle_negatives = self._sample_negative_bundle(bundle_train_ratings)
        # divide all ratings into train and test two dataframes, which consit of userId, itemId and rating three columns.
        #self.train_ratings, self.val_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings

    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings

    """def _split_loo(self, ratings):
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        val = ratings[ratings['rank_latest'] == 2]
        train = ratings[ratings['rank_latest'] > 2]
        assert train['userId'].nunique() == test['userId'].nunique() == val['userId'].nunique()
        assert len(train) + len(test) + len(val) == len(ratings)
        return train[['userId', 'itemId', 'rating']], val[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]"""

    def _sample_negative_item(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        #interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 198))
        interact_status['negative_samples'] = interact_status['negative_items'].apply(
            lambda x: random.choices(list(x), k=198)  # 允许重复
        )
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def _sample_negative_bundle(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['bundleId'].apply(set).reset_index().rename(
            columns={'bundleId': 'interacted_bundles'})
        interact_status['negative_bundles'] = interact_status['interacted_bundles'].apply(lambda x: self.bundle_pool - x)
        interact_status['negative_samples'] = interact_status['negative_bundles'].apply(
            lambda x: random.choices(list(x), k=198)  # 允许重复
        )
        #interact_status['negative_samples'] = interact_status['negative_bundles'].apply(lambda x: random.sample(x, 198))
        return interact_status[['userId', 'negative_bundles', 'negative_samples']]

    def store_all_train_data(self, num_negatives):
        """store all the train data as a list including users, items and ratings. each list consists of all users'
        information, where each sub-list stores a user's positives and negatives"""
        item_ratings = pd.merge(self.item_ratings, self.item_negatives[['userId', 'negative_items']], on='userId')
        bundle_ratings = pd.merge(self.bundle_train_ratings, self.bundle_negatives[['userId', 'negative_bundles']], on='userId')
        item_ratings['negatives'] = item_ratings['negative_items'].apply(lambda x: random.sample(x,
                                                                                                   num_negatives))  # include userId, itemId, rating, negative_items and negatives five columns.
        bundle_ratings['negatives'] = bundle_ratings['negative_bundles'].apply(lambda x: random.sample(x,
                                                                                                 num_negatives))
        # split train_ratings into groups according to userId.
        grouped_item_ratings = item_ratings.groupby('userId')
        grouped_bundle_ratings = bundle_ratings.groupby('userId')
        train_users = []
        users, items, item_ratings, bundles, bundle_ratings = [], [], [], [], []
        for userId, user_train_ratings in grouped_bundle_ratings:
            single_user = []
            user_bundle = []
            user_rating = []
            user_item = []
            user_item_rating = []
            train_users.append(userId)
            user_length = len(user_train_ratings)
            #########
            user_item_df = grouped_item_ratings.get_group(userId)
            user_item = user_item_df['itemId'].tolist()
            user_item_rating = user_item_df['rating'].tolist()
            user_item_negative = user_item_df['negatives'].tolist()
            for i in range(len(user_item)):
                for j in range(num_negatives):
                    user_item.append(user_item_negative[0][j])
                    user_item_rating.append(float(0))
            #########
            for row in user_train_ratings.itertuples():
                single_user.append(int(row.userId))
                user_bundle.append(int(row.bundleId))
                user_rating.append(float(row.rating))
                for i in range(num_negatives):
                    single_user.append(int(row.userId))
                    user_bundle.append(int(row.negatives[i]))
                    user_rating.append(float(0))  # negative samples get 0 rating

            user_item, user_item_rating = align_length(user_item, user_item_rating, user_bundle)
            assert len(single_user) == len(user_bundle) == len(user_rating)==len(user_item)==len(user_item_rating)
            assert (1 + num_negatives) * user_length == len(single_user)

            users.append(single_user)
            bundles.append(user_bundle)
            bundle_ratings.append(user_rating)
            items.append(user_item)
            item_ratings.append(user_item_rating)
        assert len(users) == len(bundles) == len(bundle_ratings) == len(self.user_pool)==len(items) == len(item_ratings)
        assert train_users == sorted(train_users)
        return [users, items,item_ratings,bundles,bundle_ratings]#要让同一个user的数据有序

    @property
    def validate_data(self):
        """create validation data"""
        val_ratings = pd.merge(self.bundle_tune_ratings, self.bundle_negatives[['userId', 'negative_samples']], on='userId')
        val_users, val_bundles, negative_users, negative_bundles = [], [], [], []
        for row in val_ratings.itertuples():
            val_users.append(int(row.userId))
            val_bundles.append(int(row.bundleId))
            for i in range(int(len(row.negative_samples) / 2)):
                negative_users.append(int(row.userId))
                negative_bundles.append(int(row.negative_samples[i]))
        assert len(val_users) == len(val_bundles)
        assert len(negative_users) == len(negative_bundles)
        assert 99 * len(val_users) == len(negative_users)
        assert val_users == sorted(val_users)
        return [torch.LongTensor(val_users), torch.LongTensor(val_bundles), torch.LongTensor(negative_users),
                torch.LongTensor(negative_bundles)]

    @property
    def test_data(self):
        """create evaluate data"""
        # return four lists, which consist userId or itemId.
        test_ratings = pd.merge(self.bundle_test_ratings, self.bundle_negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_bundles, negative_users, negative_bundles = [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_bundles.append(int(row.bundleId))
            for i in range(int(len(row.negative_samples) / 2), len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_bundles.append(int(row.negative_samples[i]))
        assert len(test_users) == len(test_bundles)
        assert len(negative_users) == len(negative_bundles)
        assert 99 * len(test_users) == len(negative_users)
        assert test_users == sorted(test_users)
        return [torch.LongTensor(test_users), torch.LongTensor(test_bundles), torch.LongTensor(negative_users),
                torch.LongTensor(negative_bundles)]