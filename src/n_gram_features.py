
import pandas as pd
import numpy as np
import os
import re 
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt 
import warnings
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
sns.set()
from tqdm import tqdm, tqdm_notebook
tqdm.pandas()

warnings.filterwarnings("ignore")
pd.options.display.max_columns = 999
plt.rcParams['figure.figsize'] = (15, 6)


size=15
params = {'legend.fontsize': 'large',
          'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'axes.titlepad': 25}
          
plt.rcParams.update(params)


def image_for_protein(molecule):
  mol = Chem.MolFromSmiles(molecule)
  d = rdMolDraw2D.MolDraw2DCairo(250, 200) # or MolDraw2DSVG to get SVGs
  mol.GetAtomWithIdx(2).SetProp('atomNote', 'foo')
  mol.GetBondWithIdx(0).SetProp('bondNote', 'bar')
  d.drawOptions().addStereoAnnotation = True
  d.drawOptions().addAtomIndices = True
  d.DrawMolecule(mol)
  d.FinishDrawing()
  with open('atom_annotation_1'+ '.jpeg', 'wb') as f:   
      f.write(d.GetDrawingText())


def get_smiles_representation(peptide):
    molecules_not_converted = []
    #Molecule from FASTA format
    try:
        molecule = Chem.MolFromFASTA(peptide)
        #SMILE representation
        smiles = Chem.MolToSmiles(molecule)
        return smiles
    except:
        molecules_not_converted.append([len(peptide), peptide])
        return molecules_not_converted


def get_string(x):
    return [char for char in x]


def get_k_mer_features(df, rel_grams):
    final_list = []
    for i, seqeunce in tqdm(enumerate(df['sequence'].tolist()), position =0 , leave=True):
      final_list.append([ df['sequence_id'][i] ] + [seqeunce.count(char.replace(" ", "")) for char in rel_grams])

    return pd.DataFrame(final_list, columns = ['sequence_id'] + ['ngram_' +str(i) for i in range(len(rel_grams))])


"""# Train and test set feature values and fasta sequence"""

train_features = pd.read_csv('/content/drive/My Drive/Genetic engineering/data/train_values.csv')
test_features = pd.read_csv('/content/drive/My Drive/Genetic engineering/data/test_values.csv')

# train_labels = pd.read_csv('/content/drive/My Drive/Genetic engineering/data/train_labels.csv')
# train_features = train_features.merge(train_labels, on = 'sequence_id', how='left')

# del train_labels
# import gc
# gc.collect()


train_features['sequence_length'] = train_features['sequence'].apply(lambda x: len(x))
train_features['sequence_length'] = train_features['sequence_length'].astype('int')
train_features['sequence_unrolled'] = train_features['sequence'].apply(lambda x: get_string(x))
train_features['sequence_unrolled'] = train_features['sequence_unrolled'].apply(lambda x: ' '.join([str(i) for i in x]))


## This gives all combinations of N-Gram features for the protein sequences ###

cvec = CountVectorizer(lowercase = False, token_pattern = " ", ngram_range = (1, 8), analyzer='char')
matrix = cvec.fit_transform(train_features.iloc[:100, ]['sequence_unrolled'].tolist())
rel_grams = cvec.get_feature_names()


train_features_v2 = get_k_mer_features(train_features, rel_grams)
test_features_v2 = get_k_mer_features(test_features, rel_grams)

##  Lab of origin for the sequences in train set ##
train_labels = pd.read_csv('/content/drive/My Drive/Genetic engineering/data/train_labels.csv')

## One Hot Encoded labels for sequences ##
label_list = []
for i in tqdm(range(train_labels.shape[0]), position = 0, leave=True):
    label_list.append([train_labels.iloc[i]['sequence_id'], np.where(train_labels.iloc[i].values[1:] == 1)[0][0]])

train_labels = pd.DataFrame(label_list, columns = ['sequence_id', 'target'])
train_features_v2 = train_features_v2.merge(train_labels, on = ['sequence_id'], how = 'left')


""" 8 Gram features"""

train_features_v2.to_csv('/content/drive/My Drive/Genetic engineering/data/train_features_v2_8g.csv', index=False)
test_features_v2.to_csv('/content/drive/My Drive/Genetic engineering/data/test_features_v2_8g.csv', index=False)
