import pandas as pd
import numpy as np
import os
import re 
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt 
import warnings
import seaborn as sns
sns.set()

from tqdm import tqdm, tqdm_notebook
tqdm.pandas()
import time
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

# Plot settings
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
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromFASTA(molecule)))
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


def get_smiles(sequence):
    global counter
    global start_time
    global train_sequence_melted
    
    if counter % 100 == 0 :
      print(f"Number of iterations completed {counter}")
      end_time = time.time()
      print(f"Time taken for process {(end_time - start_time)/60}")
      start_time = time.time()
      # train_sequence_melted.to_csv('/content/drive/My Drive/Genetic engineering/data/train_seqeunce_smile_rep_30K_60K.csv' , index = False)
    counter += 1
    return Chem.MolToSmiles(Chem.MolFromFASTA(sequence))

# Vectorized the get Smiles function    
get_smiles_vec = np.vectorize(get_smiles)