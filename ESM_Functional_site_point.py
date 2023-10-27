import argparse
import pathlib
import string
import sys, os
import torch
import torch.optim as optim
import pandas as pd
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, ShuffleSplit
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import itertools
from typing import List, Tuple
import numpy as np
import models
import torch
import torch.nn as nn

import wandb

def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from an ensemble of ESM-1v models."  # noqa
    )
    
    parser.add_argument(
        "--model-location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
        nargs="+",
    )
    
    parser.add_argument(
        "--data_input",
        type=str,
        help="Base sequence to which mutations were applied",
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    

    parser.add_argument(
        "--fingerprints",
        type=int,
        default=1,
        help="1 for keep the ESM model unchanged, 0 for fine-tuning the ESM model",
    )
    
    parser.add_argument(
        "--target_type",
        type=str,
        default="classification",
        help="Target type: ['classification', 'regression']",
    )
    
    parser.add_argument(
        "--print_res",
        type=int,
        default=0,
        help="1 for printing the results, 0 for not printing the results",
    )
    
    
    parser.add_argument(
        "--test_phase",
        type=int,
        default=0,
        help="1 for testing phase, 0 for training phase",
    )
    
    return parser

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None

    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)

def train_esm(model, batch, target, criterion, optimizer, args):
    optimizer.zero_grad()
    if args.pooling_method == "aa":
#        print(batch.shape)
        m = (batch[:, :, 0]!=0).long().cuda()
#        print("Masking shape:", m.shape)
        logits = model(batch, m)
    if args.pooling_method == "mean":
        logits = model(batch)
    if args.print_res:
        print("Train Logits:", logits)
        print("Train Target:", target, "\n")
    loss = criterion(logits, target.cuda())

    loss.backward()
    optimizer.step()
    return loss

def eval_esm(model, batch, target, criterion, args):
    if args.pooling_method == "aa":
#        print(batch.shape)
        m = (batch[:, :, 0]!=0).long().cuda()
#        print("Masking shape:", m.shape)
        logits = model(batch, m)
    if args.pooling_method == "mean":
        logits = model(batch)
    if args.print_res:
        print("Logits:", logits)
        print("Target:", target, "\n")
    loss = criterion(logits, target.cuda())
    prediction = torch.argmax(logits, dim=1)
#    print("Prediction", prediction)
#    print("Target", target)
    return loss, prediction.detach().cpu(), target
    
def main(args):
    if args.target_type == "classification":
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 2.86, 2.84, 6.8], dtype=torch.float).cuda())
        target_class=4
    elif args.target_type == "regression":
        criterion = nn.MSELoss()
        target_class=2
    else:
        raise ValueError("Target type not supported")
    batch_size = args.batch_size
    if args.target_type == "classification":
        split_method=RepeatedStratifiedKFold(n_splits=5,n_repeats=1,random_state=42)
    elif args.target_type == "regression":
        split_method=ShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
        
    df = pd.read_csv(args.data_input)
    print(df.shape)
    df = df[~df['sequence'].str.contains("_")]
    df = df[(df['abundance_score'].isna()==False)&(df['function_score'].isna()==False)]
    df.reset_index(drop=True, inplace=True)
    print(df.shape)
    
    
    # sequence = df['sequence'].apply(remove_insertions)
    
    # drop the rows in df which has '_' in the column 'sequence'
    
    sequence = df['sequence']

    mutant = df['mutation_location']
    max_length = max([len(i) for i in sequence])
    if args.target_type == "classification":
        target = df['target'].to_numpy()
    elif args.target_type == "regression":
        target = df[['abundance_score', 'function_score']].to_numpy()
    
    for model_location in args.model_location:
        esm_model, alphabet = pretrained.load_model_and_alphabet(model_location)
        
    if torch.cuda.is_available():
        esm_model = esm_model.cuda()
        print("Transferred model to GPU")
        
    batch_converter = alphabet.get_batch_converter()
    
    split_time = 0
    
    for train_index, test_index in split_method.split(sequence, target):
        linear_model = models.ESMAttention1dMean(d_embedding=1280, target_class=target_class).cuda()
        
        if args.fingerprints:
            optimizer = optim.Adam(linear_model.parameters(), lr=args.lr)
        else:
            optimizer = optim.Adam(list(esm_model.parameters())+list(linear_model.parameters()), lr=args.lr)
        
        ###
        print("Start training")
        print("batch_size:", args.batch_size)
        print("optimizer:", optimizer)
        
        ###
        print("Split:", split_time, '\n')
        split_time += 1
        
        if not args.test_phase:
            wandb.init(
                project="ESM-1v-Functional-site",
                name="{}_noPool_finetuned_{}_fingerprint_{}".format(model_location, split_time, args.fingerprints),
                entity="hirahtang",
                config={
                    "model_location": model_location,
                    "data_input": args.data_input,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "fingerprints": args.fingerprints,
                    "target_type": args.target_type,
                    "split_time": split_time
                }
                )
        
        for epoch in range(args.epochs):
            print("\tEpoch:", epoch)
            linear_model.train()
            iterate_loss = []
            for i in tqdm(range(len(train_index))[::batch_size]):  
                data = [("protein{}".format(j), sequence[j]) for j in train_index[i:i+batch_size]]
                data_y = target[train_index[i:i+batch_size]]
                data_y = torch.tensor(data_y, dtype=torch.float)
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                esm_model.eval()
                with torch.no_grad():
                    results = esm_model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)
                repre = results['representations'][33]
                repre = repre.detach().cpu()
                print(repre.shape)
                print(len(sequence[i]))
                break
            
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
   
        
    main(args)