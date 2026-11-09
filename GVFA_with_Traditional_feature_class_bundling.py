#!/usr/bin/env python
# coding: utf-8

# In[1]:


# use "env" environment 


from argparse import ArgumentParser
from models.graphcnnVSA_Binding_FULL import GraphCNN
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import Descriptors,rdMolDescriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import MolStandardize
from rdkit import Chem
from rdkit import Chem,DataStructs
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from src.create_graphs import create_graph_list
from src.embeddings import getEmbedding
from src import utilities
from src.load_data import load_data2
from src.VSA_conversion import VSA_conversion
from xgboost import XGBClassifier
from xgboost import XGBRegressor
#get_ipython().run_line_magic('matplotlib', 'inline')
#import cloudpickle
import joblib
import matplotlib.pyplot as plt
import numpy as np
import numpy as np, pandas as pd
import pandas as pd
import resource
import seaborn as sns
import sys
import torch
import warnings

warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression


parser = ArgumentParser()
parser.add_argument('-i','--infile', help='input training file')
parser.add_argument('-p','--predfile', help='input prediction file if mode == p')
parser.add_argument('-m','--method', help='ml method: rf (r) logistic regression (lr) or xgb (x)(default)')
parser.add_argument('-hvd','--HV_Dims', help='HV_Dimentions: (10k), (5k) or [100, 500, 1000, 2000, 5000] (0) (default)')
args = parser.parse_args()

infile = args.infile
predfile = args.predfile
method = args.method
HV_Dims = args.HV_Dims
sep = '\t'

if infile == None:
    parser.print_help(sys.stderr)
    sys.exit(1)

if predfile == None:
    parser.print_help(sys.stderr)
    sys.exit(1)

if method == None:
    method = 'xgb'

if method == 'x':
    method = 'xgb'

if method == 'r':
    method = 'rf'

if HV_Dims == None or HV_Dims == '0':
    HV_Dims = '0'

# # Traditional Feature extraction

# In[2]:


def make_clean_pipeline(rp_dim=1000, use_knn=False):
    # Pre-step: replace infs with NaN so imputers can handle them
    def _prep(df):
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    # Choose imputer
    imputer = KNNImputer(n_neighbors=5, weights="distance") if use_knn else SimpleImputer(strategy="median")

    pipe = Pipeline([
        ("prep",    ("passthrough")),              # placeholder; we'll call _prep manually
        ("impute",  imputer),                      # keep rows, fill NaNs
        ("clipper", RobustScaler(with_centering=True, with_scaling=True)),  # robust to outliers
        ("rp",      GaussianRandomProjection(n_components=rp_dim, random_state=42)),
    ])
    pipe._prep = _prep
    return pipe


def rename_cols(datafile):
    datafile = pd.read_csv(datafile, sep='\t', header = 0, index_col = None)

    try:
        labels = datafile['id'].values
    except:
        datafile.rename(columns={ datafile.columns[0]: "id" }, inplace = True)
        labels = datafile['id'].values
        print("/////////////////////// renaming column 1 to 'id' ///////////////////////")

    try:
        datafile.loc[datafile['target'] < 0, 'target'] = 0
    except:
        datafile.rename(columns={ datafile.columns[1]: "target" }, inplace = True)
        datafile.loc[datafile['target'] < 0, 'target'] = 0
        print("/////////////////////// renaming column 2 to 'target' ///////////////////////")

    try:
        labels = datafile['smiles'].values
    except:
        datafile.rename(columns={ datafile.columns[2]: "smiles" }, inplace = True)
        print("/////////////////////// renaming column 3 to 'smiles' ///////////////////////")
    return datafile

train_set=rename_cols(infile)
test_set=rename_cols(predfile)
print(train_set)

train_smiles_list=train_set[['smiles']]
test_smiles_list=test_set[['smiles']]

print(train_smiles_list)

### Generate 4 descriptors ....
#df4_train=utilities.generate4(train_set.smiles)
#df4_test=utilities.generate4(test_set.smiles)
### Generate 17 descriptors ....
#df17_train=utilities.generate17(train_set.smiles)
#df17_test=utilities.generate17(test_set.smiles)
### Generate 123 descriptors ....'''
#df123_train=utilities.generate123(train_set.smiles)
#df123_test=utilities.generate123(test_set.smiles)
### Generate 38 feature engineered based on the structure of the smiles ....
#df38_train=utilities.generate_features38(train_set.smiles)
#df38_test=utilities.generate_features38(test_set.smiles)
### Generate 7 funnctional groups
#df7_train=utilities.get_functional_groups(train_set.smiles)
#df7_test=utilities.get_functional_groups(test_set.smiles)
### Fingerprint 128....
#df128_train=utilities.fingerprint(train_set.smiles,2,128)
#df128_test=utilities.fingerprint(test_set.smiles,2,128)


## Prof. Ulf proposed data
print ("Computing df96 matrix ...")
df96_train=utilities.generate_desc_96(train_set.smiles)
df96_test=utilities.generate_desc_96(test_set.smiles)
#print (df96_train)
#print (df96_test)

## Prof. Ulf + chatgpt suggested data
#df193_train=utilities.generate_desc_193(train_set.smiles)
#df193_test=utilities.generate_desc_193(test_set.smiles)


# In[3]:

# #### Make hypervectors and combine  with traditional Features  - Bundling

if HV_Dims == '0':
    HV_Dimentions = [100, 500, 1000, 2000, 5000] #, 10000]
if HV_Dims == '10k':
    HV_Dimentions = [10000, 10000]
if HV_Dims == '5k':
    HV_Dimentions = [5000, 5000]
print (HV_Dimentions)

for HV_Dimention in HV_Dimentions:

    train_data, test_data = load_data2(infile, predfile)


    train_graphs = create_graph_list(train_data)
    test_graphs = create_graph_list(test_data)


    num_layers = 5
    delta_eq1 = 1
    equation_eq1 = 10
    graph_pooling_type = 'sum'  # sum, average
    neighbor_pooling_type = 'sum' # sum, average, max
    device = 1  # help='if delta is 1 will be the model with binding, if 0 model will have be without binding (default: 1)'
    device = torch.device('cpu')
    
    train_graphs = create_graph_list(train_data)
    test_graphs = create_graph_list(test_data)
    ts_graph = test_graphs.copy()
    tr_graph = train_graphs.copy()

    test_HVs = VSA_conversion(ts_graph, HV_Dimention)
    train_HVs = VSA_conversion(tr_graph, HV_Dimention)

    model_eq1 = GraphCNN(test_HVs[0].node_features.shape[1], num_layers, delta_eq1, graph_pooling_type, neighbor_pooling_type, device, equation_eq1) #.to(device)
    train_embeddings_eq1, train_labels_eq1 = getEmbedding(model_eq1, device, train_HVs)
    test_embeddings_eq1, test_labels_eq1 = getEmbedding(model_eq1, device, test_HVs)

    del train_HVs, test_HVs

    train_embeddings_eq1 = train_embeddings_eq1.squeeze(0)
    test_embeddings_eq1 = test_embeddings_eq1.squeeze(0)

    pipe = make_clean_pipeline(rp_dim=HV_Dimention, use_knn=False)

    Xtr_df = pipe._prep(df96_train).astype(np.float32)
    Xte_df = pipe._prep(df96_test).astype(np.float32)

    Xtr_rp = pipe.fit_transform(Xtr_df)   # fit on train
    Xte_rp = pipe.transform(Xte_df)       # transform test


    print(Xtr_rp.shape)
    print(train_embeddings_eq1.shape)

    A = Xtr_rp.astype(np.float32)
    A_GVFA = train_embeddings_eq1.detach().cpu().numpy().astype(np.float32)

    B = Xte_rp.astype(np.float32)
    B_GVFA = test_embeddings_eq1.detach().cpu().numpy().astype(np.float32)

    S_train = A + A_GVFA
    S_test = B + B_GVFA

    del  A, A_GVFA, B, B_GVFA

    S_train /= (np.linalg.norm(S_train, axis=1, keepdims=True) + 1e-8)
    S_test /= (np.linalg.norm(S_test, axis=1, keepdims=True) + 1e-8)

    torch.manual_seed(0)
    idx = torch.randperm(S_train.shape[0])
#    print (idx)
    print (len(idx))
    idxtr_len = int(0.8*len(idx))
    S_train_shuffled = S_train[idx]

    Xtr = S_train_shuffled[:idxtr_len]
    Xval = S_train_shuffled[idxtr_len:]
#    print ("Xtr")
#    print (Xtr)
#    print (Xval)
    print ("Xval.shape")
    print (Xval.shape)
    train_labels_eq1_shuffled = train_labels_eq1[idx]
#    print (train_labels_eq1_shuffled)
    ytr = train_labels_eq1_shuffled[:idxtr_len]
    yval = train_labels_eq1_shuffled[idxtr_len:]
    print (ytr)
    print (yval.shape)

    imethod = 0
    if method == 'xgb':
        imethod = 1
        xgb = XGBClassifier(n_estimators=2000, use_label_encoder=False, eval_metric = 'logloss',
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=42,
            n_jobs=10,#4,
            early_stopping_rounds=100,
#            early_stopping_patience = 10,
            tree_method="hist")   # fast on CPU; use "gpu_hist" if you have GPU

        print ("Running xgb.fit ...")
        xgb.fit(
            Xtr, ytr,  ########### if HYPER-PARAMTER OPTIMIZATION then Xtr, ytr MUST be split into hyperopt training set and hyperopt val set #UN
            eval_set=[(Xval, yval)],
            # early_stopping_rounds=100,
            verbose=True #False
        )


        evals_result = xgb.evals_result()
        loss = evals_result['validation_0']['logloss']

        print ("pred test")
#        print (pred)
        pred_prob = xgb.predict_proba(S_test)
#        print (pred_prob)
        pred_prob_val = xgb.predict_proba(Xval)


    if method == 'rf':
        imethod = 1
        print ("Running rf.fit ...")
        xgb = RandomForestClassifier(n_estimators = 500, n_jobs = 10)
        xgb.fit(Xtr, ytr)  ########### if HYPER-PARAMTER OPTIMIZATION then Xtr, ytr MUST be split into hyperopt training set and hyperopt val set #UN

        print ("pred test")
        pred_prob = xgb.predict_proba(S_test)
        pred_prob_val = xgb.predict_proba(Xval)

    if method == 'lr':
        imethod = 1
        print ("Running lr.fit ...")
        xgb = LogisticRegression(random_state=0)
        xgb.fit(Xtr, ytr)  ########### if HYPER-PARAMTER OPTIMIZATION then Xtr, ytr MUST be split into hyperopt training set and hyperopt val set #UN
        pred_prob = xgb.predict_proba(S_test)
        print (pred_prob)
        pred_prob_val = xgb.predict_proba(Xval)

    if imethod == 0:
        print("method not defined:", method)
        sys.exit(1)


    dftest = rename_cols(predfile)
    dftest = pd.concat([dftest, pd.DataFrame(pred_prob)], axis=1)
    dftest.columns = ['name','label','smiles', 'pred_prob0','pred_prob1']
    dftest['set'] = 'test'
    outfile = predfile + '_predict_bundling_dim' + str(HV_Dimention) + '_' + str(method) + '.csv'
    dftest.to_csv(outfile, sep='\t', index=False, header=True)
    print(dftest)

    print ("pred_prob_val")
    print (len(pred_prob_val))
    df_val = pd.DataFrame(yval)
    df_val['id'] = df_val.index
    df_val.columns = ['label','id']
    df_val = df_val[['id','label']]
    df_val['smiles'] = 'smi_nan'
    try:
        dummy = df_val['id'].values
    except:
        df_val.rename(columns={ df_val.columns[0]: "id" }, inplace = True)
        print("/////////////////////// renaming column 1 to 'id' ///////////////////////")
    df_val['id'] = df_val['id'].astype(str)
#    df_val.reset_index(drop=True, inplace=True)
#    print (df_val)
    df_val['id'] = str('val_') + df_val['id']
#    print (df_val)
    df_val = pd.concat([df_val, pd.DataFrame(pred_prob_val)], axis=1)
#    print (df_val)
    df_val.columns = ['name','label','smiles', 'pred_prob0','pred_prob1']
    df_val['set'] = 'val'
    df_val.to_csv(outfile, sep='\t', index=False, header=False, mode = 'a')
    print (df_val)


    resus = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    resus = resus/1000
    print('Max memory usage (MB): ', resus)

#    xgb_298=utilities.get_errors1(test_labels_eq1,pred_xgb,f"XGB_298 concatinate GVFA({HV_Dimention})")
#    xgb_298['Descriptors_Detail']='125 features + 128 fingerprint 7 f_group+38 fe features'
#    print(xgb_298)

