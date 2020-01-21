import numpy as np
from torch.utils.data.dataset import Dataset
from dataLoader import seq_padding, code2index, position_idx, index_seg
import torch
import pandas as pd
from torch.utils.data import DataLoader


def weightedSampling(data, classes, split):
    def make_weights_for_balanced_classes(sampled, nclasses, split):
        count = sampled.label.value_counts().to_list()
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / (float(count[i]))
        weight = [0] * int(N)
        weight_per_class[0] = weight_per_class[0] * split

        for idx, val in enumerate(sampled.label):
            weight[idx] = weight_per_class[int(val)]
        return weight

    w = make_weights_for_balanced_classes(data, classes, split)
    w = torch.DoubleTensor(w)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(w, len(w), replacement=True)
    return sampler


class HFLoader(Dataset):
    def __init__(self, token2idx, dataframe, max_len, age_vocab, code='code', age='age', mask_token=None):
        # dataframe preproecssing
        # filter out the patient with number of visits less than min_visit
        self.vocab = token2idx
        self.max_len = max_len
        self.code = dataframe[code]
        self.age = dataframe[age]
        self.label = dataframe.label
        self.age2idx = age_vocab
        self.patid = dataframe.patid
        self.mask_token=mask_token

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """
        # cut data
        age = self.age[index]
        code = self.code[index]
        label = self.label[index]
        patid = self.patid[index]

        # extract data
        age = age[(-self.max_len+1):]
        code = code[(-self.max_len+1):]

        last_age = int(age[-1])

        # avoid data cut with first element to be 'SEP'
        if code[0] != 'SEP':
            code = np.append(np.array(['CLS']), code)
            age = np.append(np.array(age[0]), age)
        else:
            code[0] = 'CLS'

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        # pad age sequence and code sequence
        age = seq_padding(age, self.max_len, token2idx=self.age2idx)

        tokens, code = code2index(code, self.vocab, self.mask_token)
#         _, label = code2index(label, self.vocab)

        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])
#         label = seq_padding(label, self.max_len, symbol=-1)

        return torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.FloatTensor([label]), torch.LongTensor([int(patid)]), \
               torch.IntTensor([int(last_age)])

    def __len__(self):
        return len(self.code)


class HF_data(object):
    def __init__(self, train_params, traindata, testdata, token2idx, age2idx, code='code', age='age'):
        self.data_positive = traindata[traindata.label == 1].reset_index()
        self.data_negative = traindata[traindata.label == 0].reset_index()
        self.test_data = testdata
        self.train_params = train_params
        self.token2idx = token2idx
        self.age2idx = age2idx
        self.code = code
        self.age = age

    def get_train_loader(self, shuffle=False):
        data = pd.concat(
            [self.data_positive, self.data_negative]
        ).reset_index()

        Dset = HFLoader(token2idx=self.token2idx, dataframe=data, max_len=self.train_params.max_len_seq,
                        age_vocab=self.age2idx, code=self.code, age=self.age)

        trainload = DataLoader(dataset=Dset,
                               batch_size=self.train_params.batch_size,
                               shuffle=shuffle,
                               num_workers=self.train_params.train_loader_workers
                               )
        return trainload

    def get_test_loader(self, shuffle=False):
        Dset = HFLoader(token2idx=self.token2idx, dataframe=self.test_data, max_len=self.train_params.max_len_seq,
                        age_vocab=self.age2idx, code=self.code, age=self.age)

        testload = DataLoader(dataset=Dset,
                              batch_size=self.train_params.batch_size,
                              shuffle=shuffle,
                              num_workers=self.train_params.test_loader_workers
                              )
        return testload

    def get_sample_train_loader(self, n_p_ratio, shuffle=False):
        data = pd.concat(
            [self.data_positive, self.data_negative.sample(n_p_ratio * len(self.data_positive))]
        ).reset_index()

        Dset = HFLoader(token2idx=self.token2idx, dataframe=data, max_len=self.train_params.max_len_seq,
                        age_vocab=self.age2idx,  code=self.code, age=self.age)

        trainload = DataLoader(dataset=Dset,
                               batch_size=self.train_params.batch_size,
                               shuffle=shuffle,
                               num_workers=self.train_params.train_loader_workers
                               )
        return trainload

    def get_sample_test_loader(self, n_p_ratio, shuffle=False):
        data_positive = self.test_data[self.test_data.label == 1].reset_index()
        data_negative = self.test_data[self.test_data.label == 0].reset_index()

        data = pd.concat(
            [data_positive, data_negative.sample(n_p_ratio * len(data_positive))]
        ).reset_index()

        Dset = HFLoader(token2idx=self.token2idx, dataframe=data, max_len=self.train_params.max_len_seq,
                        age_vocab=self.age2idx,  code=self.code, age=self.age)

        testload = DataLoader(dataset=Dset,
                              batch_size=self.train_params.batch_size,
                              shuffle=shuffle,
                              num_workers=self.train_params.test_loader_workers
                              )
        return testload

    def get_weighted_sample_train(self, split):
        # split means the ratio between positive and negative approximately if 1:4 then put 4 in split
        data = pd.concat([self.data_positive, self.data_negative]).reset_index()
        sampler = weightedSampling(data, 2, split)

        Dset = HFLoader(token2idx=self.token2idx, dataframe=data, max_len=self.train_params.max_len_seq,
                        age_vocab=self.age2idx,  code=self.code, age=self.age)

        trainload = DataLoader(dataset=Dset,
                               batch_size=self.train_params.batch_size,
                               shuffle=False,
                               num_workers=self.train_params.train_loader_workers,
                               sampler=sampler
                               )
        return trainload

    def make_loader(self, data, batch_size, shuffle=False, workers=0):
        Dset = HFLoader(token2idx=self.token2idx, dataframe=data, max_len=self.train_params.max_len_seq,
                        age_vocab=self.age2idx, code=self.code, age=self.age)

        testload = DataLoader(dataset=Dset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=workers
                              )
        return testload