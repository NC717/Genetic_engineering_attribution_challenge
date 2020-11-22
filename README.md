# This repository contains a novel solution for Genetic Engineering Attribution challenge Organized by ALT labs on Drivendata

## Challenge Overview

![Alt text](images\al-green-homepage.jpg?raw=true "")

Overview
The goal was to create an algorithm that identifies the most likely lab-of-origin for genetically engineered DNA.

Applications for genetic engineering are rapidly diversifying. Researchers across the world are using powerful new techniques in synthetic biology to solve some of the world’s most pressing challenges in medicine, agriculture, manufacturing and more. At the same time, increasingly powerful genetically engineered systems could yield unintended consequences for people, food crops, livestock, and industry. These incredible advances in capability demand tools that support accountable innovation.

Genetic engineering attribution is the process of identifying the source of a genetically engineered piece of DNA. This ability ensures that scientists who have spent countless hours developing breakthrough technology get their due credit, intellectual property is protected, and responsible innovation is promoted. By connecting a genetically engineered system with its designers, society can examine the policies, processes, and decisions that led to its creation. As has been observed in other disciplines, reducing anonymity encourages more prudent behavior within scientific and entrepreneurial communities—without stifling innovation.

Development of attribution capabilities is critical for the maturation of genetic engineering as a field, protecting the significant benefits it promises society while promoting accountability, responsibility, and dialog. In this competition, the challenge was to advance the state-of-the-art in this exciting new domain!

# Results from the hackathon leaderboard 


![Alt text](images\score_board.JPG?raw=true "")

# Final approach

To accurately predict the lab of origin for plasmid sequences a combination of features were used to create a final XGBoost classifier. The features were based in 
Graph representation learning to learn the structural information for the protein sequences, in coherence with the n-gram features which accurately capture the positional features for the sequences.

## Step 1: Conversion of protein sequences into SMILE (Simplified Molecular Input Line Entry System) notation 

I used an open source library (rdkit) to convert the protein sequences into SMILE, the final structure for a sample protein sequence is hown below.

![Alt text](images\protein_to_smile.JPG?raw=true "")

## Step 2: Final model architecture used

I tried a variety of Graph neural network based approached to build the complete model. The final model which I used was a graph attention network to learn the embeddings for the protein graphs. These embedding were then concatenated with the n-gram features and a XGB classifier was used on top of it to predict the lab of origin for the protein sequences.

![Alt text](images\final_architecture.JPG?raw=true "")
