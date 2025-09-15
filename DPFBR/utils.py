"""
    Some handy functions for pytroch model training ...
"""
import torch
import numpy as np
import copy
from sklearn.metrics import pairwise_distances
import logging
import random
import math

def group_clients(round_participant_params, similarity_matrix):
    """
    Aggregates client parameters by grouping users based on similarity.

    Args:
        round_participant_params (dict):
            A dictionary where keys are user IDs and values are dictionaries containing
            'embedding_item.weight' matrices.
        similarity_matrix (np.ndarray):
            A 2D numpy array where similarity_matrix[i][j] represents the similarity
            between user with ID participant_keys[i] and user with ID participant_keys[j].

    Returns:
        tuple:
            - groups (dict): Mapping from group_id to list of user IDs in the group.
            - param (dict): Mapping from group_id to the averaged 'embedding_item.weight' matrix.
    """
    # print("group")
    # Extract the list of participant user IDs
    participant_keys = list(round_participant_params.keys())
    num_participants = len(participant_keys)
    # Create mappings from user IDs to indices and vice versa
    user_to_idx = {user_id: idx for idx, user_id in enumerate(participant_keys)}
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    # Initialize the set of users that are yet to be grouped
    remaining_users = set(participant_keys)
    # Initialize dictionaries to store groups and their aggregated parameters
    groups = {}
    param = {}
    user_to_group = {}
    # Initialize group ID counter
    group_id = 0
    while remaining_users:
        # Randomly select a user from the remaining users
        user = random.choice(list(remaining_users))
        user_idx = user_to_idx[user]
        # Retrieve similarity scores for the selected user to all other users
        similarities = similarity_matrix[user_idx]
        # Calculate the average similarity excluding self-similarity
        # To exclude self-similarity, set it to -inf or remove it from the calculation
        # Here, we'll compute the mean excluding the user's own similarity
        other_similarities = np.delete(similarities, user_idx)
        avg_similarity = np.mean(other_similarities)
        # Identify users with similarity >= average similarity
        similar_users = [
            idx_to_user[idx]
            for idx, sim in enumerate(similarities)
            if sim <= avg_similarity and idx_to_user[idx] in remaining_users
        ]
        # If no other users meet the criteria, ensure at least the selected user is in the group
        if not similar_users:
            similar_users = [user]
        # Assign the group to the groups dictionary
        groups[group_id] = similar_users
        # 将用户到组的映射记录下来
        for u in similar_users:
            user_to_group[u] = group_id
        # Extract the 'embedding_item.weight' matrices for all users in the group
        weights = [
            round_participant_params[u]['embedding_item.weight']
            for u in similar_users
        ]
        # 堆叠张量并计算平均值
        stacked_weights = torch.stack(weights)  # 形状: (组内用户数, ...)
        avg_weight = torch.mean(stacked_weights, dim=0)
        # Store the averaged weights in the param dictionary
        param[group_id] = avg_weight
        # Remove the grouped users from the remaining_users set
        remaining_users -= set(similar_users)
        # Increment the group ID for the next group
        group_id += 1

    print("分组数量:",group_id+1)
    #print(user_to_group)
    return param, user_to_group,group_id+1


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                     lr=params['lr'],
                                     weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


def construct_user_relation_graph_via_item(round_user_params, item_num, latent_dim, similarity_metric):
    # prepare the item embedding array.
    item_embedding = np.zeros((len(round_user_params), item_num * latent_dim), dtype='float32')
    for user in round_user_params.keys():
        item_embedding[user] = round_user_params[user]['embedding_item.weight'].numpy().flatten()
    # construct the user relation graph.
    adj = pairwise_distances(item_embedding, metric=similarity_metric)
    if similarity_metric == 'cosine':
        return adj
    else:
        return -adj


def select_topk_neighboehood(user_realtion_graph, neighborhood_size, neighborhood_threshold):
    topk_user_relation_graph = np.zeros(user_realtion_graph.shape, dtype='float32')
    if neighborhood_size > 0:
        for user in range(user_realtion_graph.shape[0]):
            user_neighborhood = user_realtion_graph[user]
            topk_indexes = user_neighborhood.argsort()[-neighborhood_size:][::-1]
            for i in topk_indexes:
                topk_user_relation_graph[user][i] = 1/neighborhood_size
    else:
        similarity_threshold = np.mean(user_realtion_graph)*neighborhood_threshold
        for i in range(user_realtion_graph.shape[0]):
            high_num = np.sum(user_realtion_graph[i] > similarity_threshold)
            if high_num > 0:
                for j in range(user_realtion_graph.shape[1]):
                    if user_realtion_graph[i][j] > similarity_threshold:
                        topk_user_relation_graph[i][j] = 1/high_num
            else:
                topk_user_relation_graph[i][i] = 1

    return topk_user_relation_graph


def MP_on_graph(round_user_params, item_num, latent_dim, topk_user_relation_graph, layers):
    # prepare the item embedding array.
    item_embedding = np.zeros((len(round_user_params), item_num*latent_dim), dtype='float32')
    for user in round_user_params.keys():
        item_embedding[user] = round_user_params[user]['embedding_item.weight'].numpy().flatten()

    # aggregate item embedding via message passing.
    aggregated_item_embedding = np.matmul(topk_user_relation_graph, item_embedding)
    for layer in range(layers-1):
        aggregated_item_embedding = np.matmul(topk_user_relation_graph, aggregated_item_embedding)

    # reconstruct item embedding.
    item_embedding_dict = {}
    for user in round_user_params.keys():
        item_embedding_dict[user] = torch.from_numpy(aggregated_item_embedding[user].reshape(item_num, latent_dim))
    item_embedding_dict['global'] = sum(item_embedding_dict.values())/len(round_user_params)
    return item_embedding_dict


def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def compute_regularization(model, parameter_label):
    reg_fn = torch.nn.MSELoss(reduction='mean')
    for name, param in model.named_parameters():
        if name == 'embedding_bundle.weight':
            reg_loss = reg_fn(param, parameter_label)
            return reg_loss

def compute_item_regularization(model, parameter_label):
    reg_fn = torch.nn.MSELoss(reduction='mean')
    for name, param in model.named_parameters():
        if name == 'embedding_item.weight':
            reg_loss = reg_fn(param, parameter_label)
            return reg_loss
