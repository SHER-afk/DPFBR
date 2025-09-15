import torch
from torch.autograd import Variable


from utils import *
from metrics import MetronAtK
import random
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
from torch.distributions.laplace import Laplace


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        # self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        # self._writer.add_text('config', str(config), 0)
        #self.server_model_param = {}
        self.param = {}
        self.users_to_group = {}
        self.client_model_params = {}
        self.server_model_params={}
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()
        self.top_k = 10

    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        item_rating_tensor=torch.FloatTensor(user_train_data[2]),
                                        bundle_tensor=torch.LongTensor(user_train_data[3]),
                                        bundle_rating_tensor=torch.FloatTensor(user_train_data[4]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def fed_train_single_batch(self, model_client, batch_data,bundle_item, optimizers,round_id):
        """train a batch and return an updated model."""
        users, items, item_ratings,bundles,bundle_ratings = batch_data[0], batch_data[1], batch_data[2],batch_data[3],batch_data[4]
        item_ratings,bundle_ratings = item_ratings.float(),bundle_ratings.float()
        params_dict = model_client.state_dict()

        reg_item_embedding = copy.deepcopy(params_dict['embedding_item.weight'].data)
        reg_bundle_embedding = copy.deepcopy(params_dict['embedding_bundle.weight'].data)

        if self.config['use_cuda']:
            users, items, item_ratings, bundles, bundle_ratings = (
                users.cuda(), items.cuda(), item_ratings.cuda(), bundles.cuda(), bundle_ratings.cuda()
            )
            reg_item_embedding = reg_item_embedding.cuda()
            reg_bundle_embedding = reg_bundle_embedding.cuda()

        # 将 bundle_item 预处理为 tensor 索引对（bundleId, itemId）
        bundle_ids = torch.tensor(bundle_item['bundleId'].values, dtype=torch.long)
        item_ids = torch.tensor(bundle_item['itemId'].values, dtype=torch.long)

        if self.config['use_cuda']:
            bundle_ids = bundle_ids.cuda()
            item_ids = item_ids.cuda()

        # 获取每个 itemId 的 embedding（shape: [N, embed_dim]）
        #print(item_ids.size(),item_ids)
        """max_idx = reg_item_embedding.shape[0]
        print("✅ reg_item_embedding.shape:", reg_item_embedding.shape)
        print("✅ item_ids.max():", item_ids.max().item())
        print("✅ item_ids.min():", item_ids.min().item())

        # 检查是否越界
        invalid_mask = item_ids >= max_idx
        if invalid_mask.any():
            print("❌ 越界 item_id 数量:", invalid_mask.sum().item())
            print("示例越界 item_ids:", item_ids[invalid_mask][:10].tolist())
            # 你可以选择 raise 或 filter
            raise ValueError("item_ids 中存在越界索引！")"""

        item_embeds = reg_item_embedding[item_ids]  # shape: [N_items, embed_dim]

        # 利用 scatter_add 来聚合每个 bundle 的 embedding 之和
        num_bundles = reg_bundle_embedding.shape[0]
        embed_dim = reg_item_embedding.shape[1]
        """invalid_bundle_mask = bundle_ids >= num_bundles
        if invalid_bundle_mask.any():
            print("❌ 越界的 bundle_id 数量:", invalid_bundle_mask.sum().item())
            print("示例越界 bundle_ids:", bundle_ids[invalid_bundle_mask][:10].tolist())
            raise ValueError(f"bundle_ids 中存在越界索引！（最大应为 {num_bundles - 1}）")"""
        bundle_embed_sum = torch.zeros((num_bundles, embed_dim), device=reg_item_embedding.device)
        bundle_item_count = torch.zeros(num_bundles, device=reg_item_embedding.device)

        # scatter item_embeds 加到对应 bundle 的位置
        #print(bundle_ids, item_embeds)
        bundle_embed_sum.index_add_(0, bundle_ids, item_embeds)
        bundle_item_count.index_add_(0, bundle_ids, torch.ones_like(bundle_ids, dtype=torch.float))

        # 避免除以 0
        bundle_item_count[bundle_item_count == 0] = 1.0

        #计算每个 bundle 的平均 embedding
        reg_bundle_embedding = bundle_embed_sum / bundle_item_count.unsqueeze(1)

        optimizer, optimizer_u, optimizer_i,optimizer_b = optimizers

        optimizer.zero_grad()
        optimizer_u.zero_grad()
        optimizer_i.zero_grad()
        optimizer_b.zero_grad()
        ratings_item,ratings_bundle = model_client(items,bundles)
        loss_item = self.crit(ratings_item.view(-1), item_ratings)
        loss_bundle = self.crit(ratings_bundle.view(-1), bundle_ratings)
        mean_item = torch.mean(loss_item)
        loss=self.config['lamda_1']*mean_item+loss_bundle
        assert torch.isfinite(loss)
        if round_id != 0:
            regularization_term = compute_regularization(model_client, reg_bundle_embedding)
            loss += self.config['reg'] * regularization_term
        loss.backward()
        optimizer.step()
        optimizer_u.step()
        optimizer_i.step()
        optimizer_b.step()
        return model_client, loss.item()

    def aggregate_clients_params(self, round_item_params,round_bundle_params):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        t = 0
        for user in round_bundle_params.keys():
            # load a user's parameters.
            user_params = round_bundle_params[user]
            # print(user_params)
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
            else:
                for key in user_params.keys():
                    self.server_model_param[key].data += user_params[key].data
            t += 1
        for key in self.server_model_param.keys():
            self.server_model_param[key].data = self.server_model_param[key].data / len(round_bundle_params)

        # construct the user relation graph via embedding similarity.
        similarity_matrics = construct_user_relation_graph_via_item(round_item_params, self.config['num_items'],
                                                            self.config['latent_dim'],
                                                            self.config['similarity_metric'])
        self.param,self.users_to_group,group_num=group_clients(round_item_params,similarity_matrics)
        return group_num


    def fed_train_a_round(self, all_train_data,bundle_item, round_id):
        """train a round."""
        # sample users participating in single round.
        num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
        participants = random.sample(range(self.config['num_users']), num_participants)
        # store users' model parameters of current round.
        round_item_params ,round_bundle_params= {},{}

        # perform model updating for each participated user.
        for user in participants:
            # copy the client model architecture from self.model
            model_client = copy.deepcopy(self.model)
            # for the first round, client models copy initialized parameters directly.
            # for other rounds, client models receive updated user embedding and aggregated item embedding from server
            # and use local updated mlp parameters from last round.
            if round_id != 0:
                user_param_dict = copy.deepcopy(self.model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()#这个里面记录了上一轮本地训练的结果，这样传进去才能收敛，才是接着上一轮继续训练
                #下面这句用来传群组内的avg
                group_id = self.users_to_group.get(user, None)
                if group_id is not None:
                    user_param_dict['embedding_item.weight'] = copy.deepcopy(self.param[group_id]).cuda()
                user_param_dict['embedding_bundle.weight'] = copy.deepcopy(
                    self.server_model_param['embedding_bundle.weight'].data).cuda()
                model_client.load_state_dict(user_param_dict)
            # Defining optimizers
            # optimizer is responsible for updating mlp parameters.
            optimizer = torch.optim.SGD(
                model_client.affine_output.parameters(),
                lr=self.config['lr'])  # MLP optimizer
            # optimizer_u is responsible for updating user embedding.
            optimizer_u = torch.optim.SGD(model_client.embedding_user.parameters(),
                                          lr=self.config['lr'] / self.config['clients_sample_ratio'] * self.config[
                                              'lr_eta'] - self.config['lr'])  # User optimizer
            # optimizer_i is responsible for updating item embedding.
            optimizer_i = torch.optim.SGD(model_client.embedding_item.parameters(),
                                          lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'] -
                                             self.config['lr'])  # Item optimizer
            optimizer_b = torch.optim.SGD(model_client.embedding_bundle.parameters(),
                                          lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'] -
                                             self.config['lr'])  # Item optimizer
            optimizers = [optimizer, optimizer_u, optimizer_i,optimizer_b]
            # load current user's training data and instance a train loader.
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user],all_train_data[3][user],all_train_data[4][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            # update client model.
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client, loss = self.fed_train_single_batch(model_client, batch,bundle_item, optimizers, round_id)
            # print('[User {}]'.format(user))
            # obtain client model parameters.
            client_param = model_client.state_dict()
            # store client models' user embedding using a dict.
            self.client_model_params[user] = copy.deepcopy(client_param)
            for key in self.client_model_params[user].keys():
                self.client_model_params[user][key] = self.client_model_params[user][key].data.cpu()
            # round_participant_params[user] = copy.deepcopy(self.client_model_params[user])
            # del round_participant_params[user]['embedding_user.weight']
            round_item_params[user],round_bundle_params[user]={},{}
            round_item_params[user]['embedding_item.weight'] = self.client_model_params[user]['embedding_item.weight']
            round_bundle_params[user]['embedding_bundle.weight'] =self.client_model_params[user]['embedding_bundle.weight']

            round_bundle_params[user]['embedding_bundle.weight'] += Laplace(0, self.config['dp']).expand(
                round_bundle_params[user]['embedding_bundle.weight'].shape).sample()
            round_item_params[user]['embedding_item.weight'] += Laplace(0, self.config['dp']).expand(
            round_item_params[user]['embedding_item.weight'].shape).sample()

        # aggregate client models in server side.
        group_num=self.aggregate_clients_params(round_item_params,round_bundle_params)
        return 0,group_num

    def fed_evaluate(self, evaluate_data):
        test_users, test_bundles = evaluate_data[0], evaluate_data[1]
        negative_users, negative_bundles = evaluate_data[2], evaluate_data[3]

        if self.config['use_cuda']:
            test_users = test_users.cuda()
            test_bundles = test_bundles.cuda()
            negative_users = negative_users.cuda()
            negative_bundles = negative_bundles.cuda()

        test_scores = []
        test_user_list = []
        test_bundle_list = []

        negative_scores = []
        neg_user_list = []
        neg_bundle_list = []

        all_loss = {}

        user_num = self.config['num_users']

        for user in range(user_num):
            user_model = copy.deepcopy(self.model)
            user_param_dict = copy.deepcopy(self.model.state_dict())
            if user in self.client_model_params:
                for key in self.client_model_params[user]:
                    user_param_dict[key] = self.client_model_params[user][key].data.cuda()
            user_model.load_state_dict(user_param_dict)
            user_model.eval()

            with torch.no_grad():
                # 获取该 user 所有正样本 index
                user_pos_idx = (test_users == user).nonzero(as_tuple=True)[0]
                pos_user = test_users[user_pos_idx]
                pos_bundle = test_bundles[user_pos_idx]

                # 模型预测正样本
                _, pos_score = user_model(pos_bundle, pos_bundle)

                test_scores.append(pos_score)
                test_user_list.append(pos_user)
                test_bundle_list.append(pos_bundle)

                # 每个正样本对应 99 个负样本，找到 user 的所有负样本位置
                neg_user_idx = (negative_users == user).nonzero(as_tuple=True)[0]
                neg_user = negative_users[neg_user_idx]
                neg_bundle = negative_bundles[neg_user_idx]

                _, neg_score = user_model(neg_bundle, neg_bundle)

                negative_scores.append(neg_score)
                neg_user_list.append(neg_user)
                neg_bundle_list.append(neg_bundle)

        # 拼接所有结果
        test_scores = torch.cat(test_scores)
        test_users = torch.cat(test_user_list)
        test_bundles = torch.cat(test_bundle_list)

        negative_scores = torch.cat(negative_scores)
        negative_users = torch.cat(neg_user_list)
        negative_bundles = torch.cat(neg_bundle_list)

        if self.config['use_cuda']:
            test_users = test_users.cpu()
            test_bundles = test_bundles.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_bundles = negative_bundles.cpu()
            negative_scores = negative_scores.cpu()

        """# 调试检查长度是否一致
        print(len(test_users), len(test_bundles), len(test_scores))
        print(len(negative_users), len(negative_bundles), len(negative_scores))"""

        # 传给 Metron
        self._metron.subjects = [
            test_users.tolist(),
            test_bundles.tolist(),
            test_scores.tolist(),
            negative_users.tolist(),
            negative_bundles.tolist(),
            negative_scores.tolist()
        ]

        hit_ratio = self._metron.cal_hit_ratio()
        ndcg = self._metron.cal_ndcg()

        return hit_ratio, ndcg, all_loss

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)
