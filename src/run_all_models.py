from Plasmid_to_smile import *
from xgb_and_evaluation_metric import *

## Import the relevant dependencies as required ##

'''
This portion of the code converts the protein sequences into SMILES
*- Since the train data size was huge, proteins were converted into SMILES in subsets
'''


config = {
        'raw_train_path', '/content/drive/My Drive/Genetic engineering/data/train_seqeunce.csv',
         'raw_test_path', '/content/drive/My Drive/Genetic engineering/data/test_seqeunce.csv',
         'n_gram_train_path', '',
         'n_gram_test_path', '',
         'graph_features_train', '',
         'graph_features_test', ''
         }


train_sequence_melted = pd.read_csv(config['train_path'])
test_sequence_melted = pd.read_csv(config['test_path'])

counter = 0 
start_time = time.time()


train_sequence_melted['sequence_length'] = train_sequence_melted['clipped_sequence'].apply(lambda x: len(x))
train_sequence_melted = train_sequence_melted.iloc[30000:60000, ]
train_sequence_melted.shape

# Will have to change the column names as provided in the dataset used for sequence classification
def save_smile_df(train_features):
  smile_list = []
  for i in tqdm_notebook(range(train_features.shape[0])):
    smile_list.append([train_features.iloc[i]['sequence_id'], train_features.iloc[i]['clipped_sequence'], get_smiles(train_features.iloc[i]['clipped_sequence'])])
    if i %100==0:
      pd.DataFrame(smile_list, columns = ['seqeunce_id', 'seqeunce', 'smile']).to_csv('/content/drive/My Drive/Genetic engineering/data/train_seqeunce_smile_rep_30K_60K.csv' , index = False)

# Saves the proteins converted to SMILES iteratively
save_smile_df(train_sequence_melted)

## Conversion completed  ##

'''
Model Training Steps

'''

'''
Step 1 -: Read in the N-gram features created from the script 'n_gram_features.py'
Step 2 -: Train the GAT model from the script '' 
Step 3 -: Read in the graph represenations created for every sequence in the train set

'''
# N-GRam features #
n_gram_train = pd.read_csv(config['n_gram_train_path'])
n_gram_test = pd.read_csv(config['n_gram_test_path'])

# Graph embeddings 512 dimensional #
graph_train = pd.read_csv(config['n_gram_train_path'])
graph_test = pd.read_csv(config['n_gram_test_path'])

# Merge the features from n-grams and Gram embeddings 512

all_feats_train = n_gram_train.merge(graph_train, on='sequence_id', how = 'left')
all_feats_test = n_gram_test.merge(graph_test, on='sequence_id', how = 'left')

## Select the relevant features for training the final XGB model ##
best_cols = [col for col in all_feats_train.columns if col not in train_labels.columns.tolist() + ['sequence_id', 'sequence', 'target']]
print(len(best_cols))

## OHE the labels for train set ##
train_labels = pd.read_csv('/content/drive/My Drive/Genetic engineering/data/train_labels.csv')
label_list = []
for i in tqdm(range(train_labels.shape[0]), position = 0, leave=True):
    label_list.append([train_labels.iloc[i]['sequence_id'], np.where(train_labels.iloc[i].values[1:] == 1)[0][0]])

## Relevant data for XGB Classifier ##
all_feats_train['target'] = label_list 
predictors = all_feats_train[best_cols]
predictors_test = all_feats_test[best_cols]
target = all_feats_train[['target']]

## 5-Fold Stratified model for final Prediction for the sequence labels ##
n_folds = 5
kf = StratifiedKFold(n_splits = n_folds)
cv_scores = []
pred_full_test_xgb = 0

pred_train = np.zeros((predictors.shape))
pred_test = np.zeros((predictors_test.shape))

for dev_index, val_index in kf.split(predictors, target):
    dev_X, val_X = predictors.iloc[dev_index], predictors.iloc[val_index]
    dev_y, val_y = target.iloc[dev_index], target.iloc[val_index]
    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, predictors_test)
    pred_full_test_xgb = pred_full_test_xgb + pred_test_y
    pred_train[val_index] = pred_val_y

#     # label_val_y = np.argmax(pred_val_y, axis = 1)
#     print("ROC_AUC_FOR_THIS_ITERATION_IS :: -- >> ", f1_score(val_y, label_val_y, average = 'weighted'))
#     print("\n")
#     cv_scores.append(f1_score(val_y, label_val_y, average = 'weighted'))
# np.mean(cv_scores)

## Get the predictions from the XGB model or use the probabilities from the pred_full_test_xgb/n_folds as the probabilities for the  test sequences ##
submission = all_feats_test['sequence_id'].to_list()
submission = pred_full_test_xgb/n_folds
submission.to_csv('./output/submission_xgb.csv', index=False)
