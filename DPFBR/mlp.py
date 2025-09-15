import torch
from engine import Engine
from utils import use_cuda, resume_checkpoint


class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.num_bundles=config['num_bundles']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_bundle= torch.nn.Embedding(num_embeddings=self.num_bundles, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim*2, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices,bundle_indices):
        user_embedding = self.embedding_user(torch.LongTensor([0 for i in range(len(item_indices))]).cuda())
        item_embedding = self.embedding_item(item_indices)
        vector_item = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        """for idx, _ in enumerate(range(len(self.fc_layers))):
            vector_item = self.fc_layers[idx](vector_item)
            vector_item = torch.nn.ReLU()(vector_item)"""
        logits_item = self.affine_output(vector_item)
        rating_item = self.logistic(logits_item)

        user_embedding = self.embedding_user(torch.LongTensor([0 for i in range(len(bundle_indices))]).cuda())
        bundle_embedding=self.embedding_bundle(bundle_indices)
        vector_bundle = torch.cat([user_embedding, bundle_embedding], dim=-1)  # the concat latent vector
        """for idx, _ in enumerate(range(len(self.fc_layers))):
            vector_bundle = self.fc_layers[idx](vector_bundle)
            vector_bundle = torch.nn.ReLU()(vector_bundle)"""
        logits_bundle = self.affine_output(vector_bundle)
        rating_bundle = self.logistic(logits_bundle)
        return rating_item, rating_bundle

    def init_weight(self):
        pass


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = MLP(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(MLPEngine, self).__init__(config)
        print(self.model)
