from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import numpy as np
import torch
from torch import nn

class GaussianModel():
    def __init__(self,):
        super().__init__()
    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.cov = np.cov(data, rowvar=0)
        self.MVN = multivariate_normal(self.mean, self.cov, allow_singular = True)
    def decision_function(self, data):
        neg_log_prob = -self.MVN.logpdf(data)
        # log_prob[log_prob>10]=10
        return neg_log_prob

class IndependantGaussianModel():
    def __init__(self,):
        super().__init__()
    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.var = data.var(axis=0)
        self.dim = len(self.mean)
    def decision_function(self, data):
        err = data - self.mean
        a = self.dim*(np.log(2*np.pi))/2
        b = np.log(self.var).sum()/2
        c = ((err**2)/self.var).sum(axis=1)/2
        return a+b+c

class MahalanobisDistance():
    def __init__(self,):
        super().__init__()
    def fit(self, data):
        self.pca = PCA(whiten=True)
        self.pca = self.pca.fit(data)
    def decision_function(self, data):
        err = self.pca.transform(data)
        return (err **2).mean(-1)

#%%
class RandomMapLinear(nn.Module):
    def __init__(self, input_dim, out_dim = 500, seed = 0):
        super().__init__()
        matrix = np.random.rand(input_dim, out_dim)
        col_sum = matrix.sum(axis=0)
        np.random.seed(seed)
        self.random_map = matrix / col_sum[np.newaxis,:]
    def forward(self, x):
        score = np.matmul(x, self.random_map)
        return score

    def decision_function(self, x):
        score = self(x)
        return score.detach().numpy()


class RandomMapperV3(nn.Module):
    def __init__(self, input_dim, n_random = 500):
        super().__init__()
        self.input_dim = input_dim       
        self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Sigmoid(), 
                nn.Linear(128, 1))
        self.n_random = n_random
    def forward(self, x):
        with torch.no_grad():
            score = self.model(x)            
        return score
    def decision_function(self, x):
        score_list = []
        for i in range(self.n_random):
            self.initializing(seed=i)
            score = self(x)
            score_list.append(score.detach().numpy())
        return np.concatenate(score_list,axis=-1)

    def initializing(self, seed = 0):
        torch.manual_seed(seed)
        for m in self.model:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

class OneShotRandomMapper(nn.Module):
    def __init__(self, input_dim, n_random = 500, seed = 0):
        super().__init__()
        self.input_dim = input_dim       
        self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Sigmoid(), 
                nn.Linear(128, n_random))
        self.n_random = n_random
        self.seed = seed
        self.initializing(seed = self.seed)

    def forward(self, x):
        with torch.no_grad():
            score = self.model(x)
        return score

    def decision_function(self, x):
        score = self.model(x)
        return score.detach().numpy()

    def initializing(self, seed = 0):
        torch.manual_seed(seed)
        for m in self.model:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

#%% 
class DropRandomMapperV2(nn.Module):
    def __init__(self, input_dim, n_random = 500, seed = 0, dr_rates = 0.5):
        super().__init__()
        self.input_dim = input_dim 
        self.dr = dr_rates      
        self.model = nn.Sequential(
                ModifiedDropout(p=self.dr, d_in = self.input_dim),
                nn.Linear(input_dim, 128),
                nn.Sigmoid(), 
                ModifiedDropout(p=self.dr, d_in = 128),
                nn.Linear(128, 1))
        self.n_random = n_random
        self.seed = seed
        self.weight_initializing(seed = self.seed)

    def forward(self, x):
        with torch.no_grad():
            score = self.model(x)
        return score

    def decision_function(self, x):
        score_list = []

        for i in range(self.n_random):
            score = self(x)
            self.dropout_initializing(seed = i)
            score_list.append(score.detach().numpy())
        return np.concatenate(score_list,axis=-1)

    def weight_initializing(self, seed = 0):
        torch.manual_seed(seed)
        for m in self.model:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def dropout_initializing(self, seed = 0):
        torch.manual_seed(seed)
        for m in self.model:
            if hasattr(m, 'reset_dr_mask'):
                m.reset_dr_mask()


class ModifiedDropout(nn.Module):
    def __init__(self, d_in, p = 0.5):
        super().__init__()
        self.p = p
        self.premask = torch.ones(d_in)*p
        self.mask = torch.bernoulli(self.premask)

    def forward(self, x):
        return torch.mul(x, self.mask)

    def reset_dr_mask(self):
        self.mask = torch.bernoulli(self.premask)

class DropRandomMapper(nn.Module):
    def __init__(self, input_dim, n_random = 500, seed = 0, dr_rates = 0.5):
        super().__init__()
        self.input_dim = input_dim 
        self.dr = dr_rates      
        self.model = nn.Sequential(
                nn.Dropout(self.dr),
                nn.Linear(input_dim, 128),
                nn.Sigmoid(), 
                nn.Dropout(self.dr),
                nn.Linear(128, 1))
        self.n_random = n_random
        self.seed = seed
        self.initializing(seed = self.seed)

    def forward(self, x):
        with torch.no_grad():
            score = self.model(x)
        return score

    def decision_function(self, x):
        score_list = []
        self.model.train()
        for i in range(self.n_random):
            score = self(x)
            score_list.append(score.detach().numpy())
        return np.concatenate(score_list,axis=-1)

    def initializing(self, seed = 0):
        torch.manual_seed(seed)
        for m in self.model:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

class DropNShotRandomMapper(nn.Module):
    def __init__(self, input_dim, n_random = 500, seed = 0, dr_rates = 0.5, drop_time = 5):
        super().__init__()
        self.input_dim = input_dim 
        self.dr = dr_rates      
        self.drop_time = drop_time
        self.model = nn.Sequential(
                nn.Dropout(self.dr),
                nn.Linear(input_dim, 128),
                nn.Sigmoid(), 
                nn.Dropout(self.dr),
                nn.Linear(128, 1))
        self.n_random = n_random
        self.seed = seed
        self.initializing(seed = self.seed)

    def forward(self, x):
        with torch.no_grad():
            score = self.model(x)
        return score

    def decision_function(self, x):
        score_list = []
        self.model.train()
        for i in range(self.n_random//self.drop_time):
            self.initializing(seed=i)
            for j in range(self.drop_time):
                score = self(x)
                score_list.append(score.detach().numpy())    
        return np.concatenate(score_list,axis=-1)

    def initializing(self, seed = 0):
        torch.manual_seed(seed)
        for m in self.model:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

#%%
class DropNShotRandomMapperV2(nn.Module):
    def __init__(self, input_dim, n_random = 500, seed = 0, dr_rates = 0.5, drop_time = 5):
        super().__init__()
        self.input_dim = input_dim 
        self.dr = dr_rates      
        self.drop_time = drop_time
        self.model = nn.Sequential(
                ModifiedDropout(p = self.dr, d_in = self.input_dim),
                nn.Linear(input_dim, 128),
                nn.Sigmoid(), 
                ModifiedDropout(p = self.dr, d_in = 128),
                nn.Linear(128, n_random//drop_time))
        self.n_random = n_random
        self.seed = seed
        self.weight_initializing(seed = self.seed)

    def forward(self, x):
        with torch.no_grad():
            score = self.model(x)
        return score

    def decision_function(self, x):
        score_list = []
        for j in range(self.drop_time):
            score = self(x)
            self.dropout_initializing(seed = j )
            score_list.append(score.detach().numpy())    
        return np.concatenate(score_list,axis=-1)

    def weight_initializing(self, seed = 0):
        torch.manual_seed(seed)
        for m in self.model:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def dropout_initializing(self, seed = 0):
        torch.manual_seed(seed)
        for m in self.model:
            if hasattr(m, 'reset_dr_mask'):
                m.reset_dr_mask()
