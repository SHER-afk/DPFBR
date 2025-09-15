import pandas as pd
import numpy as np
import datetime
import os

from matplotlib import pyplot as plt


import argparse
from mlp import MLPEngine
from data import SampleGenerator
from utils import *
import warnings
from matplotlib.ticker import MaxNLocator

# 忽略所有警告
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='bundle')
parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
parser.add_argument('--clients_sample_num', type=int, default=0)
parser.add_argument('--num_round', type=int, default=50)
parser.add_argument('--local_epoch', type=int, default=1)
parser.add_argument('--neighborhood_size', type=int, default=0)
parser.add_argument('--neighborhood_threshold', type=float, default=1.)
parser.add_argument('--mp_layers', type=int, default=1)
parser.add_argument('--similarity_metric', type=str, default='cosine')
parser.add_argument('--reg', type=float, default=0.01)
parser.add_argument('--lr_eta', type=int, default=80)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--dataset', type=str, default='Youshu')
parser.add_argument('--num_users', type=int)
parser.add_argument('--num_items', type=int)
parser.add_argument('--num_bundles', type=int)
parser.add_argument('--latent_dim', type=int, default=16)
parser.add_argument('--num_negative', type=int, default=4)
parser.add_argument('--layers', type=str, default='64, 32, 16, 8')
parser.add_argument('--l2_regularization', type=float, default=0.)
parser.add_argument('--lamda_1', type=float, default=0.01)
parser.add_argument('--dp', type=float, default=0.1)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--device_id', type=int, default=1)
parser.add_argument('--model_dir', type=str, default='checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model')
parser.add_argument('--construct_graph_source', type=str, default='item')
args = parser.parse_args()

# Model.
config = vars(args)
if len(config['layers']) > 1:
    config['layers'] = [int(item) for item in config['layers'].split(',')]
else:
    config['layers'] = int(config['layers'])
if config['dataset'] == 'Youshu':
    config['num_users'] = 2959
    config['num_items'] = 32770
    config['num_bundles'] = 4771#4771
elif config['dataset'] == 'food':
    config['num_users'] =277
    config['num_items'] =1482
    config['num_bundles'] = 800
elif config['dataset'] == 'iFashion':
    config['num_users'] =3103
    config['num_items'] = 31885
    config['num_bundles'] = 24838
elif config['dataset'] == 'NetEase':
    config['num_users'] = 2870
    config['num_items'] = 30001
    config['num_bundles'] = 30001
else:
    pass
engine = MLPEngine(config)

# Logging.
path = 'log/'
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logname = os.path.join(path, current_time+'.txt')
initLogging(logname)

# Load Data
dataset_dir = "data/" + config['dataset']
if config['dataset'] == "Youshu":
    bundle_train_rating = pd.read_csv(dataset_dir+ "/" + "user_bundle_train_reindexed.txt", sep=' ', header=None, names=['uid', 'bid'], engine='python')
    bundle_train_rating['rating'] = 1
    bundle_test_rating = pd.read_csv(dataset_dir+ "/" + "user_bundle_test_reindexed.txt", sep=' ', header=None, names=['uid', 'bid'], engine='python')
    bundle_test_rating['rating'] = 1
    bundle_tune_rating = pd.read_csv(dataset_dir+ "/" +"user_bundle_tune_reindexed.txt", sep=' ', header=None, names=['uid', 'bid'], engine='python')
    bundle_tune_rating['rating'] = 1
    item_rating = pd.read_csv(dataset_dir+ "/" +"cut_user_item_reindexed.txt", sep=' ', header=None, names=['uid', 'mid'], engine='python')
    item_rating['rating'] = 1
    bundle_item = pd.read_csv(dataset_dir+ "/" +"bundle_item.txt", sep='	', header=None, names=['bundleId', 'itemId'], engine='python')

elif config['dataset'] == "food":
    bundle_train_rating = pd.read_csv(dataset_dir + "/" + "user_bundle_train.csv", sep=',', header=None,
                                      names=['uid', 'bid'], engine='python')
    bundle_train_rating['rating'] = 1
    bundle_test_rating = pd.read_csv(dataset_dir + "/" + "user_bundle_test.csv", sep=',', header=None,
                                     names=['uid', 'bid'], engine='python')
    bundle_test_rating['rating'] = 1
    bundle_tune_rating = pd.read_csv(dataset_dir + "/" + "user_bundle_tune.csv", sep=',', header=None,
                                     names=['uid', 'bid'], engine='python')
    bundle_tune_rating['rating'] = 1
    item_rating = pd.read_csv(dataset_dir + "/" + "user_item_mapped.csv", sep=',', header=None, names=['uid', 'mid'],
                              engine='python')
    item_rating['rating'] = 1
    bundle_item = pd.read_csv(dataset_dir + "/" + "bundle_item_mapped.csv", sep=',', header=None,
                              names=['bundleId', 'itemId'], engine='python')
    #print(bundle_test_rating.shape, bundle_train_rating.shape, bundle_tune_rating.shape)


elif config['dataset'] == "iFashion":
    bundle_train_rating = pd.read_csv(dataset_dir + "/" + "filtered_user_bundle_train.txt", sep='	', header=None,
                                      names=['uid', 'bid'], engine='python')
    bundle_train_rating['rating'] = 1
    bundle_test_rating = pd.read_csv(dataset_dir + "/" + "filtered_user_bundle_test.txt", sep='	', header=None,
                                     names=['uid', 'bid'], engine='python')
    bundle_test_rating['rating'] = 1
    bundle_tune_rating = pd.read_csv(dataset_dir + "/" + "filtered_user_bundle_tune.txt", sep='	', header=None,
                                     names=['uid', 'bid'], engine='python')
    bundle_tune_rating['rating'] = 1
    item_rating = pd.read_csv(dataset_dir + "/" + "ranged_user_item.txt", sep='	', header=None, names=['uid', 'mid'],
                              engine='python')
    item_rating['rating'] = 1
    bundle_item = pd.read_csv(dataset_dir + "/" + "limited_bundle_item.txt", sep='	', header=None,
                              names=['bundleId', 'itemId'], engine='python')
    #print(bundle_test_rating.shape,bundle_train_rating.shape,bundle_tune_rating.shape)

elif config['dataset'] == "NetEase":
    bundle_train_rating = pd.read_csv(dataset_dir + "/" + "filtered_user_bundle_train.txt", sep='	', header=None,
                                      names=['uid', 'bid'], engine='python')
    bundle_train_rating['rating'] = 1
    bundle_test_rating = pd.read_csv(dataset_dir + "/" + "filtered_user_bundle_test.txt", sep='	', header=None,
                                     names=['uid', 'bid'], engine='python')
    bundle_test_rating['rating'] = 1
    bundle_tune_rating = pd.read_csv(dataset_dir + "/" + "filtered_user_bundle_tune.txt", sep='	', header=None,
                                     names=['uid', 'bid'], engine='python')
    bundle_tune_rating['rating'] = 1
    item_rating = pd.read_csv(dataset_dir + "/" + "filtered_user_item.txt", sep='	', header=None, names=['uid', 'mid'],
                              engine='python')
    item_rating['rating'] = 1
    bundle_item = pd.read_csv(dataset_dir + "/" + "bundle_item_mapped.txt", sep='	', header=None,
                              names=['bundleId', 'itemId'], engine='python')

else:
    pass

# Reindex
user_id = item_rating[['uid']].drop_duplicates()
user_id['userId'] = np.arange(len(user_id))
item_rating = pd.merge(item_rating, user_id, on=['uid'], how='left')
item_id = item_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
item_rating = pd.merge(item_rating, item_id, on=['mid'], how='left')
item_rating = item_rating[['userId', 'itemId', 'rating']]


user_id = bundle_train_rating[['uid']].drop_duplicates()
user_id['userId'] = np.arange(len(user_id))
bundle_train_rating = pd.merge(bundle_train_rating, user_id, on=['uid'], how='left')
bundle_id = bundle_train_rating[['bid']].drop_duplicates()
bundle_id['bundleId'] = np.arange(len(bundle_id))
bundle_train_rating = pd.merge(bundle_train_rating, bundle_id, on=['bid'], how='left')
bundle_train_rating = bundle_train_rating[['userId', 'bundleId', 'rating']]

user_id = bundle_test_rating[['uid']].drop_duplicates()
user_id['userId'] = np.arange(len(user_id))
bundle_test_rating = pd.merge(bundle_test_rating, user_id, on=['uid'], how='left')
bundle_id = bundle_test_rating[['bid']].drop_duplicates()
bundle_id['bundleId'] = np.arange(len(bundle_id))
bundle_test_rating = pd.merge(bundle_test_rating, bundle_id, on=['bid'], how='left')
bundle_test_rating = bundle_test_rating[['userId', 'bundleId', 'rating']]

user_id = bundle_tune_rating[['uid']].drop_duplicates()
user_id['userId'] = np.arange(len(user_id))
bundle_tune_rating = pd.merge(bundle_tune_rating, user_id, on=['uid'], how='left')
bundle_id = bundle_tune_rating[['bid']].drop_duplicates()
bundle_id['bundleId'] = np.arange(len(bundle_id))
bundle_tune_rating = pd.merge(bundle_tune_rating, bundle_id, on=['bid'], how='left')
bundle_tune_rating = bundle_tune_rating[['userId', 'bundleId', 'rating']]



logging.info('Range of userId is [{}, {}]'.format(item_rating.userId.min(), item_rating.userId.max()))
logging.info('Range of itemId is [{}, {}]'.format(item_rating.itemId.min(), item_rating.itemId.max()))

# DataLoader for training
sample_generator = SampleGenerator(item_ratings=item_rating,bundle_train_ratings=bundle_train_rating,bundle_test_ratings=bundle_test_rating,bundle_tune_ratings=bundle_tune_rating)
validate_data = sample_generator.validate_data
test_data = sample_generator.test_data

hit_ratio_list = []
ndcg_list = []
val_hr_list = []
val_ndcg_list = []
train_loss_list = []
test_loss_list = []
val_loss_list = []
best_val_hr = 0
final_test_round = 0
group_nums=[]
for round in range(config['num_round']):
    # break
    logging.info('-' * 80)
    logging.info('Round {} starts !'.format(round))

    all_train_data = sample_generator.store_all_train_data(config['num_negative'])
    logging.info('-' * 80)
    logging.info('Training phase!')
    #tr_loss,group_num = engine.fed_train_a_round(all_train_data,round_id=round)
    tr_loss, group_num = engine.fed_train_a_round(all_train_data, bundle_item, round_id=round)
    # break
    group_nums.append(group_num)
    train_loss_list.append(tr_loss)

    logging.info('-' * 80)
    logging.info('Testing phase!')
    hit_ratio, ndcg, te_loss = engine.fed_evaluate(test_data)
    test_loss_list.append(te_loss)
    # break
    logging.info('[Testing Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(round, hit_ratio, ndcg))
    hit_ratio_list.append(hit_ratio)
    ndcg_list.append(ndcg)

    logging.info('-' * 80)
    logging.info('Validating phase!')
    val_hit_ratio, val_ndcg, v_loss = engine.fed_evaluate(validate_data)
    val_loss_list.append(v_loss)
    logging.info(
        '[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(round, val_hit_ratio, val_ndcg))
    val_hr_list.append(val_hit_ratio)
    val_ndcg_list.append(val_ndcg)

    if val_hit_ratio >= best_val_hr:
        best_val_hr = val_hit_ratio
        final_test_round = round


current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
rounds = range(config['num_round'])
# 图1：Hit Ratio 和 NDCG
plt.figure(figsize=(10, 5))
plt.plot(rounds, hit_ratio_list, label='Hit Ratio', color='#FF69B4', linestyle='-', linewidth=3)
plt.plot(rounds, ndcg_list, label='NDCG', color='#ff7f0e', linestyle='-', linewidth=3)
plt.title("NDCG and Hit Ratio")
plt.xlabel("Round")
plt.ylabel("Value")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 图2：Group Number
plt.figure(figsize=(10, 5))
plt.plot(rounds, group_nums, color='#800080', linestyle='-', linewidth=3)
plt.title("Group Number")
plt.xlabel("Round")
plt.ylabel("Value")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(loc='best')
plt.tight_layout()
plt.show()
str = current_time + '-' + 'layers: ' + str(config['layers']) + '-' + 'lr: ' + str(config['lr']) + '-' + \
      'clients_sample_ratio: ' + str(config['clients_sample_ratio']) + '-' + 'num_round: ' + str(config['num_round']) \
      + '-' 'neighborhood_size: ' + str(config['neighborhood_size']) + '-' + 'mp_layers: ' + str(config['mp_layers']) \
      + '-' + 'negatives: ' + str(config['num_negative']) + '-' + 'lr_eta: ' + str(config['lr_eta']) + '-' + \
      'batch_size: ' + str(config['batch_size']) + '-' + 'hr: ' + str(hit_ratio_list[final_test_round]) + '-' \
      + 'ndcg: ' + str(ndcg_list[final_test_round]) + '-' + 'best_round: ' + str(final_test_round) + '-' + \
      'similarity_metric: ' + str(config['similarity_metric']) + '-' + 'neighborhood_threshold: ' + \
      str(config['neighborhood_threshold']) + '-' + 'reg: ' + str(config['reg'])
file_name = "sh_result/"+'-'+config['dataset']+".txt"
with open(file_name, 'a') as file:
    file.write(str + '\n')

logging.info('fedgraph')
logging.info('clients_sample_ratio: {}, lr_eta: {}, bz: {}, lr: {}, dataset: {}, layers: {}, negatives: {}, '
             'neighborhood_size: {}, neighborhood_threshold: {}, mp_layers: {}, similarity_metric: {}'.
             format(config['clients_sample_ratio'], config['lr_eta'], config['batch_size'], config['lr'],
                    config['dataset'], config['layers'], config['num_negative'], config['neighborhood_size'],
                    config['neighborhood_threshold'], config['mp_layers'], config['similarity_metric']))

logging.info('hit_list: {}'.format(hit_ratio_list))
logging.info('ndcg_list: {}'.format(ndcg_list))
logging.info('Best test hr: {}, ndcg: {} at round {}'.format(hit_ratio_list[final_test_round],
                                                             ndcg_list[final_test_round],
                                                             final_test_round))
