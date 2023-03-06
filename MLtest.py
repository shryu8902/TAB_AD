#%%
import datetime, os, sys, socket, pandas as pd, pytorch_lightning as pl
from pyod.models.gmm import GMM
from Model.RandomModels import *
from base import *
from data_loader import *
from utils import *

#%%
args = get_args()
args['PATH_DATA_INFO_SET'] = './Data/data_info_set.json'
args['PATH_DATASETS'] = './Data'
args['N_WORKERS'] = 0
#%%
data_info_set = get_data_info_set(data_info_set_path='./Data/data_info_set.json')
dataset_list = list(data_info_set.keys())

for key in data_info_set:
    if 'normal' not in data_info_set[key]['label_dict'].values():
        dataset_list.remove(key)
args['DATASETS'] = dataset_list
args['INPUT_SCALES'] = ['SS']
args['SEEDS']=[0,1,2]

exp_results = []
for seed in args['SEEDS']:
    args['SEED'] = seed
    for dataset in args['DATASETS']:
        for input_scale in args['INPUT_SCALES']:
            args['INPUT_SCALE'] = input_scale
            args['DATASET'] = dataset

            data_info, loaders = load_dataset(args)
            # case, case_key = get_case(args=args, key_return=True)

            X_train, Y_train = (x.numpy() for x in loaders['train'].dataset[:])
            X_valid, Y_valid = (x.numpy() for x in loaders['valid'].dataset[:])
            X_test, Y_test = (x.numpy() for x in loaders['test'].dataset[:])

            # RM_DROP = DropRandomMapper(input_dim=args['RAW_DIM'], n_random= 500, seed = args['SEED'], dr_rates = 0.5)
            RM_DROP_V2 = DropRandomMapperV2(input_dim=args['RAW_DIM'], n_random= 500, seed = args['SEED'], dr_rates = 0.5)

            RM_1SHOT = OneShotRandomMapper(input_dim=args['RAW_DIM'], n_random= 500, seed = args['SEED'])
            # RM_DNS = DropNShotRandomMapper(input_dim=args['RAW_DIM'], n_random= 500, seed = args['SEED'], dr_rates = 0.5, drop_time=5)
            RM_DNS_V2 = DropNShotRandomMapperV2(input_dim=args['RAW_DIM'], n_random= 500, seed = args['SEED'], dr_rates = 0.5, drop_time=5)

            RM = RandomMapperV3(input_dim = args['RAW_DIM'], n_random= 500)
            LRM = RandomMapLinear(input_dim = args['RAW_DIM'], out_dim = 500, seed = args['SEED'])
            RAW = 'RAW'

            # model_dict = {'RM_DROP':RM_DROP, 'RM_1SHOT':RM_1SHOT, 'RM_DNS':RM_DNS, 'RM':RM, 'LRM':LRM, 'RAW':RAW}
            model_dict = {'RM_DROP_V2':RM_DROP_V2, 'RM_DNS_V2':RM_DNS_V2}

            for i, (key, model) in enumerate(model_dict.items()):
                model_name = key                
                if key == 'RAW':
                    Y_train_hat = X_train
                    Y_test_hat = X_test
                else:
                    Y_train_hat = model.decision_function(torch.Tensor(X_train))
                    Y_test_hat = model.decision_function(torch.Tensor(X_test))

                scorer = MahalanobisDistance()
                scorer.fit(Y_train_hat)
                score = scorer.decision_function(Y_test_hat)

                if (score == np.inf).any():
                    auc, prauc = 0, 0
                else:
                    auc = auc_calc(Y_test, score, int(args['LABELS_ABNORMAL']))
                    prauc = prauc_calc(Y_test, score, int(args['LABELS_ABNORMAL']))

                exp_results.append({'seed':seed, 'data':dataset, 'input_scale':input_scale,
                                    'model':model_name, 'scoring': 'MD',
                                    'auc':auc, 'prauc':prauc})

df_results = pd.DataFrame(exp_results)
# df_results.to_csv('./results_with_RS.csv') scaled with robust scaler
# df_results.to_csv('./0303_results2.csv') # randomization scheme comparison
df_results.to_csv('./0306_results.csv') # fix dropout model


#%%
data_info_set = get_data_info_set(data_info_set_path='./Data/data_info_set.json')
dataset_list = list(data_info_set.keys())

for key in data_info_set:
    if 'normal' not in data_info_set[key]['label_dict'].values():
        dataset_list.remove(key)
args['DATASETS'] = dataset_list
args['INPUT_SCALES'] = ['SS']
args['SEEDS']=[0,1,2]

exp_results = []
for seed in args['SEEDS']:
    args['SEED'] = seed
    for dataset in args['DATASETS']:
        for input_scale in args['INPUT_SCALES']:
            args['INPUT_SCALE'] = input_scale
            args['DATASET'] = dataset

            data_info, loaders = load_dataset(args)
            # case, case_key = get_case(args=args, key_return=True)

            X_train, Y_train = (x.numpy() for x in loaders['train'].dataset[:])
            X_valid, Y_valid = (x.numpy() for x in loaders['valid'].dataset[:])
            X_test, Y_test = (x.numpy() for x in loaders['test'].dataset[:])

            RM_DROP = DropRandomMapper(input_dim=args['RAW_DIM'], n_random= 500, seed = args['SEED'], dr_rates = 0.5)
            RM = RandomMapperV3(input_dim = args['RAW_DIM'], n_random= 500)
            LRM = RandomMapLinear(input_dim = args['RAW_DIM'], out_dim = 500, seed = args['SEED'])
            model_dict = {'NRM':RM, 'LRM':LRM}
            for i, (key, model) in enumerate(model_dict.items()):
                model_name = key                
                Y_train_hat = model.decision_function(torch.Tensor(X_train))
                Y_test_hat = model.decision_function(torch.Tensor(X_test))
                for scoring in ['MGM', 'IMGM', 'MD', 'GMIX', 'MGMinRAW']:
                    if scoring == 'MGM' or scoring == 'MGMinRAW':
                        scorer = GaussianModel()
                    elif scoring == 'IGM':
                        scorer = IndependantGaussianModel()
                    elif scoring == 'GMIX':
                        scorer = GMM(n_components = 1)
                    elif scoring == 'MD':
                        scorer = MahalanobisDistance()

                    if scoring == 'MGMinRAW':
                        scorer.fit(X_train)
                        score = scorer.decision_function(X_test)
                    else:
                        scorer.fit(Y_train_hat)
                        score = scorer.decision_function(Y_test_hat)

                    if (score == np.inf).any():
                        auc, prauc = 0, 0
                    else:
                        auc = auc_calc(Y_test, score, int(args['LABELS_ABNORMAL']))
                        prauc = prauc_calc(Y_test, score, int(args['LABELS_ABNORMAL']))

                    exp_results.append({'seed':seed, 'data':dataset, 'input_scale':input_scale,
                                        'model':model_name, 'scoring': scoring,
                                        'auc':auc, 'prauc':prauc})

df_results = pd.DataFrame(exp_results)
# df_results.to_csv('./results_with_RS.csv') scaled with robust scaler
df_results.to_csv('./0303_results.csv')
#%%
