'''
This code use 7 features of atoms as node features. 
USe "sol" python environment 
'''


from src.create_graphs import create_graph_list
from src.load_data import load_undata, df_read_csv #,load_data
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN
import torch
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    # import mean_squared_error, mean_absolute_error, r2_score
import resource
import sys
from argparse import ArgumentParser
import pandas as pd
#import cloudpickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


parser = ArgumentParser()
parser.add_argument('-i','--infile', help='input training file')
parser.add_argument('-p','--predfile', help='input prediction file')
parser.add_argument('-m','--method', help='ml method: rf (r) logistic regression (lr) or xgb (x)(default)')
parser.add_argument('-hvd','--HV_Dims', help='HV_Dimentions: (10k) or [100, 500, 1000, 2000, 5000] (0) (default)')
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


train_data = load_undata(infile, task ='train')
print (train_data)

val_data = load_undata(infile, task ='val')
print (val_data)

test_data = load_undata(predfile, 'test')
print (test_data)


train_graphs = create_graph_list(train_data)
val_graphs = create_graph_list(val_data)
test_graphs = create_graph_list(test_data)



num_layers = 5
delta_eq1 = 1
equation_eq1 = 10
graph_pooling_type = 'sum'  # sum, average
neighbor_pooling_type = 'sum' # sum, average, max
device = 1  # help='if delta is 1 will be the model with binding, if 0 model will have be without binding (default: 1)'
device = torch.device('cpu')

if HV_Dims == '0':
    dims = [100, 500, 1000, 2000, 5000] #, 10000]
if HV_Dims == '10k':
    dims = [10000, 10000]
print (dims)

for dim in dims:
    print ("dim:", dim)
#    train_graphs = create_graph_list(train_data)
#    test_graphs = create_graph_list(test_data)
    ts_graph = test_graphs.copy()
#    print (ts_graph)
    tr_graph = train_graphs.copy()
#    print (tr_graph)
    val_graph = val_graphs.copy()
    test_HVs = VSA_conversion(ts_graph, dim)
    train_HVs = VSA_conversion(tr_graph, dim)
    val_HVs = VSA_conversion(val_graph, dim)

    model_eq1 = GraphCNN(test_HVs[0].node_features.shape[1], num_layers, delta_eq1, graph_pooling_type, neighbor_pooling_type, device, equation_eq1) #.to(device)
    train_embeddings_eq1, train_labels_eq1 = getEmbedding(model_eq1, device, train_HVs)
    val_embeddings_eq1, val_labels_eq1 = getEmbedding(model_eq1, device, val_HVs)
    test_embeddings_eq1, test_labels_eq1 = getEmbedding(model_eq1, device, test_HVs)

    train_embeddings_eq1 = train_embeddings_eq1.squeeze(0)
    val_embeddings_eq1 = val_embeddings_eq1.squeeze(0)
    test_embeddings_eq1 = test_embeddings_eq1.squeeze(0)

    # print(len(test_graphs))
    # print(test_graphs[0].node_features)

    print(train_embeddings_eq1)
    print(train_labels_eq1)
    print(train_embeddings_eq1.shape, train_labels_eq1.shape)

    imethod = 0
    if method == 'xgb':
        imethod = 1
        xgb = XGBClassifier(n_estimators=2000, use_label_encoder=False, eval_metric = 'logloss',
            #params
              learning_rate=0.03,
              max_depth=7,
              subsample=0.8,
              colsample_bytree=0.8,
              reg_lambda=1.0,
              reg_alpha=0.0,
            #params end
            random_state=42,
            n_jobs=10,#4,
            early_stopping_rounds=100,
#            early_stopping_patience = 10,
            tree_method="hist")   # fast on CPU; use "gpu_hist" if you have GPU

        print ("Running xgb.fit ...")
        xgb.fit(
            train_embeddings_eq1, train_labels_eq1,   ########### if HYPER-PARAMTER OPTIMIZATION then train_embeddings_eq1, train_labels_eq1 MUST be split into hyperopt training set and hyperopt val set #UN
            eval_set=[(val_embeddings_eq1, val_labels_eq1)],
            #early_stopping_rounds=100,
            verbose=True #False
        )

        evals_result = xgb.evals_result()
        loss = evals_result['validation_0']['logloss']

        print (loss)

    # ---------- Evaluate ----------

#    pred = xgb.predict(test_embeddings_eq1)
#    rmse = mean_squared_error(test_labels_eq1, pred)
#    mae  = mean_absolute_error(test_labels_eq1, pred)
#    r2   = r2_score(test_labels_eq1, pred)
#    print(f"Dimention,{dim},MAE,{mae},RMSE,{rmse},R2,{r2}")

        print ("pred test")
#        print (pred)
        pred_prob = xgb.predict_proba(test_embeddings_eq1)
        print (pred_prob)
        pred_prob_val = xgb.predict_proba(val_embeddings_eq1)

    if method == 'rf':
        imethod = 1
        print ("Running rf.fit ...")
        xgb = RandomForestClassifier(n_estimators = 500, n_jobs = 10)
        xgb.fit(train_embeddings_eq1, train_labels_eq1)    ########### if HYPER-PARAMTER OPTIMIZATION then train_embeddings_eq1, train_labels_eq1 MUST be split into hyperopt training set and hyperopt val set #UN
        pred_prob = xgb.predict_proba(test_embeddings_eq1)
        print (pred_prob)
        pred_prob_val = xgb.predict_proba(val_embeddings_eq1)
 
    if method == 'lr':
        imethod = 1
        print ("Running lr.fit ...")
        xgb = LogisticRegression(random_state=0)
        xgb.fit(train_embeddings_eq1, train_labels_eq1)    ########### if HYPER-PARAMTER OPTIMIZATION then train_embeddings_eq1, train_labels_eq1 MUST be split into hyperopt training set and hyperopt val set #UN
        pred_prob = xgb.predict_proba(test_embeddings_eq1)
        print (pred_prob)
        pred_prob_val = xgb.predict_proba(val_embeddings_eq1)

    if imethod == 0:
        print("method not defined:", method)
        sys.exit(1)

    dftest = df_read_csv(predfile)
    dftest = pd.concat([dftest, pd.DataFrame(pred_prob)], axis=1)
    dftest.columns = ['name','label','smiles', 'pred_prob0','pred_prob1']
    dftest['set'] = 'test'
    outfile = predfile + '_predict_dim' + str(dim) + '_' + str(method) + '.csv'
    dftest.to_csv(outfile, sep='\t', index=False, header=True)

    print ("pred_prob_val")
    print (len(pred_prob_val))
    dftrain = df_read_csv(infile)
    dftrain = dftrain.sample(frac=1, random_state=42)
    nrows = dftrain.shape[0]
    nrowstr = int(0.8*nrows)
    df_val = dftrain[nrowstr:]
    df_val.reset_index(drop=True, inplace=True)
    print (df_val)
    try:
        dummy = df_val['id'].values
    except:
        df_val.rename(columns={ df_val.columns[0]: "id" }, inplace = True)
        print("/////////////////////// renaming column 1 to 'id' ///////////////////////")
    df_val['id'] = df_val['id'].astype(str)
    df_val['id'] = str('val_') + df_val['id']
    print (df_val)
    df_val = pd.concat([df_val, pd.DataFrame(pred_prob_val)], axis=1)
    df_val.columns = ['name','label','smiles', 'pred_prob0','pred_prob1']
    df_val['set'] = 'val'
    df_val.to_csv(outfile, sep='\t', index=False, header=False, mode = 'a')


    resus = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    resus = resus/1000
    print('Max memory usage (MB): ', resus)

    del dftest
    del df_val
    del dftrain
    del xgb
    del train_embeddings_eq1
    del val_embeddings_eq1
    del test_embeddings_eq1
    del test_labels_eq1
    del train_labels_eq1
    del val_labels_eq1
    del model_eq1
    del test_HVs
    del train_HVs
    del val_HVs
