import torch.nn as nn
import pytorch_pretrained_bert as Bert
import numpy as np
import torch
from model import Embedding
import math
import gpytorch
from model.utils import prior


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config, feature_dict):
        super(BertEmbeddings, self).__init__()
        # self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
        # self.segment_embeddings = Embedding(config.seg_vocab_size, config.hidden_size)
        # self.age_embeddings = Embedding(config.age_vocab_size, config.hidden_size)
        # self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
        #     from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))
        #
        # self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.feature_dict = feature_dict

        if self.feature_dict['word']:
            self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, prior=prior.Normal(shape=(config.vocab_size, config.hidden_size), scale=torch.ones((config.vocab_size, config.hidden_size)) * config.prior_rate))
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        if self.feature_dict['seg']:
            self.segment_embeddings = Embedding(config.seg_vocab_size, config.hidden_size)
        else:
            self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)
        if self.feature_dict['age']:
            self.age_embeddings = Embedding(config.age_vocab_size, config.hidden_size)
        else:
            self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)

        self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

        if self.feature_dict['norm']:
            self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, word_ids, age_ids=None, seg_ids=None, posi_ids=None, age=True):
        kl = 0
        if self.feature_dict['word']:
            word_embed, word_kl = self.word_embeddings(word_ids)
            kl = kl + word_kl
        else:
            word_embed = self.word_embeddings(word_ids)

        if self.feature_dict['seg']:
            segment_embed, seg_kl = self.segment_embeddings(seg_ids)
            kl = kl + seg_kl
        else:
            segment_embed = self.segment_embeddings(seg_ids)

        if self.feature_dict['age']:
            age_embed, age_kl = self.age_embeddings(age_ids)
            kl = kl + age_kl
        else:
            age_embed = self.age_embeddings(age_ids)

        posi_embeddings = self.posi_embeddings(posi_ids)

        if age:
            embeddings = word_embed + segment_embed + age_embed + posi_embeddings
        else:
            embeddings = word_embed + segment_embed + posi_embeddings

        if self.feature_dict['norm']:
            embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)
        return embeddings, kl

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)

class BertPooler(nn.Module):
    def __init__(self, config, n_dim):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, n_dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertModel(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, n_dim, feature_dict):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config=config, feature_dict=feature_dict)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = BertPooler(config, n_dim)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)
        if seg_ids is None:
            seg_ids = torch.zeros_like(input_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output, kl = self.embeddings(input_ids, age_ids, seg_ids, posi_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output, kl


class GaussianProcessLayer(gpytorch.models.AbstractVariationalGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64, ard_num_dims=1):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_size=num_dim
        )
        variational_strategy = gpytorch.variational.AdditiveGridInterpolationVariationalStrategy(
            self, grid_size=grid_size, grid_bounds=[grid_bounds], num_dim=num_dim,
            variational_distribution=variational_distribution
        )
        super(GaussianProcessLayer, self).__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class BertHF(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, n_dim, grid_bounds=(-10., 10.), grid_size=128, ard_num_dims=1, feature_dict=None, cuda1='cpu:0', cuda2='cpu:0', split=True):
        super(BertHF, self).__init__(config)
        self.cuda1=cuda1
        self.cuda2=cuda2
        self.split=split

        if feature_dict is None:
            feature_dict = {
                'word': True,
                'age': True,
                'seg': True,
                'norm': True
            }

        self.bert = BertModel(config, n_dim, feature_dict)
        # self.num_labels = num_labels

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gp_layer = GaussianProcessLayer(num_dim=n_dim, grid_size=grid_size, grid_bounds=grid_bounds,
                                             ard_num_dims=ard_num_dims)
        self.grid_bounds = grid_bounds

    # self.loss = BayesLoss(nn.BCEWithLogitsLoss())

        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None, labels=None, beta=1):
        _, pooled_output, kl = self.bert(input_ids, age_ids, seg_ids, posi_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output = self.dropout(pooled_output)
        pooled_output = gpytorch.utils.grid.scale_to_bounds(pooled_output, self.grid_bounds[0], self.grid_bounds[1])

        # if self.split:
        #     pooled_output.to(self.cuda2)

        res = self.gp_layer(pooled_output.to(self.cuda2))
        kl = kl.to(self.cuda2)

        return res, kl

    def allocateGPU(self):
        self.bert.to(self.cuda1)
        self.gp_layer.to(self.cuda2)