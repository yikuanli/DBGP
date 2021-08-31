import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert
import numpy as np
import gpytorch
import math
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, age_ids=None, seg_ids=None, posi_ids=None, age=True):
        if seg_ids is None:
            seg_ids = torch.zeros_like(word_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(word_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(word_ids)

        word_embed = self.word_embeddings(word_ids)
        segment_embed = self.segment_embeddings(seg_ids)
        age_embed = self.age_embeddings(age_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)

        if age:
            embeddings = word_embed + segment_embed + age_embed + posi_embeddings
        else:
            embeddings = word_embed + segment_embed + posi_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

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
    def __init__(self, config, n_dim):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config=config)
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

        embedding_output = self.embeddings(input_ids, age_ids, seg_ids, posi_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class GaussianProcessLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing, mean_type='constant'):
        # induce_init = torch.distributions.Normal(torch.zeros(grid_size, n_dim), torch.ones(grid_size,n_dim)).sample()
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super(GaussianProcessLayer, self).__init__(variational_strategy)

        if mean_type == 'constant':
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = gpytorch.means.LinearMean(input_dims)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

        self.linear_layer = nn.Linear(input_dims, 1)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class GPClassifier(DeepGP):
    def __init__(self, input_dims, middle_size, num_inducing, num_layer):
        if num_layer == 1:
            gp_layer = GaussianProcessLayer(
                input_dims=input_dims,
                output_dims=None,
                num_inducing=num_inducing,
                mean_type='constant'
            )
            self.gp_layer = gp_layer
        elif num_layer == 2:
            first_layer = GaussianProcessLayer(
                input_dims=input_dims,
                output_dims=middle_size,
                num_inducing=num_inducing,
                mean_type='linear',
            )
            second_layer = GaussianProcessLayer(
                input_dims=first_layer.output_dims,
                output_dims=None,
                num_inducing=num_inducing,
                mean_type='constant'
            )
            self.gp_layer = nn.ModuleList([first_layer, second_layer])
        elif num_layer == 3:
            first_layer = GaussianProcessLayer(
                input_dims=input_dims,
                output_dims=middle_size,
                num_inducing=num_inducing,
                mean_type='linear',
            )
            second_layer = GaussianProcessLayer(
                input_dims=first_layer.output_dims,
                output_dims=middle_size,
                num_inducing=num_inducing,
                mean_type='linear'
            )
            third_layer = GaussianProcessLayer(
                input_dims=first_layer.output_dims,
                output_dims=None,
                num_inducing=num_inducing,
                mean_type='constant'
            )
            self.gp_layer = nn.ModuleList([first_layer, second_layer, third_layer])
        else:
            raise ValueError("GP layer should be no more than 3")
        super().__init__()

    def forward(self, inputs):
        output = self.gp_layer(inputs)
        return output


class BertHF(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, n_dim, num_inducing, middle_size, num_layer):
        super(BertHF, self).__init__(config)
        # self.num_labels = num_labels
        self.bert = BertModel(config, n_dim=n_dim)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.gp_layer = GPClassifier(n_dim, middle_size, num_inducing, num_layer)

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, age_ids, seg_ids, posi_ids, attention_mask, output_all_encoded_layers=False)
        res = self.gp_layer(pooled_output)
        return res
