import argparse
import pathlib
import string
import sys, os
import torch
import torch.optim as optim
import pandas as pd
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


from torch.utils.data import Dataset, DataLoader


import wandb

class CustomDataset:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    deletekeys['_'] = None

    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)

def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from features extracted from the paper of cagiada."  # noqa
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
    
    
    return parser

def main(args):
    if args.target_type == "classification":
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 2.86, 2.84, 6.8], dtype=torch.float).cuda())
        target_class=4
    elif args.target_type == "regression":
        criterion = nn.MSELoss()
        target_class=2
    else:
        raise ValueError("Target type not supported")
    
    if args.target_type == "classification":
        split_method=RepeatedStratifiedKFold(n_splits=5,n_repeats=1,random_state=42)
    elif args.target_type == "regression":
        split_method=ShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
    df = pd.read_csv(args.data_input)
    print(df.shape)
    df = df[(df['abundance_score'].isna()==False)&(df['function_score'].isna()==False)&(df['0'].isna()==False)&(df['1'].isna()==False)&(df['2'].isna()==False)&(df['3'].isna()==False)&(df['4'].isna()==False)&(df['5'].isna()==False)&(df['6'].isna()==False)&(df['7'].isna()==False)]
    df.reset_index(drop=True, inplace=True)
    print(df.shape)
    
    if args.target_type == "classification":
        target = df['target'].to_numpy()
    elif args.target_type == "regression":
        target = df[['abundance_score', 'function_score']].to_numpy()
        
    X = df[['0', '1', '2', '3', '4', '5', '6', '7']].to_numpy()
    
    split_time = 0
    
    for train_index, test_index in split_method.split(X, target):
        linear_model = models.ESMAttention1dMean(d_embedding=8, target_class=target_class).cuda()
        
        optimizer = optim.Adam(linear_model.parameters(), lr=args.lr)
        
        # Transfer the X and target with train_index and test_index to train_dataloder and test_dataloader
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = target[train_index], target[test_index]
        # transfer the X_train and Y_train to train_dataloader
        train_dataset = CustomDataset(X_train, Y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # transfer the X_test and Y_test to test_dataloader
        test_dataset = CustomDataset(X_test, Y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        
        # WanDB Init
        wandb.init(
            project="ESM-1v-Functional-site",
            name="cagiada_finetuned_{}".format(split_time),
            entity="hirahtang",
            config={
                "data_input": args.data_input,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "target_type": args.target_type,
                "split_time": split_time
            }
            )
        split_time += 1
        for epoch in range(args.epochs):
            print("\tEpoch: ", epoch)
            linear_model.train()
            train_loss_list = []
            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad()

                batch_x = batch[0]
                batch_y = batch[1]
                # Transfer batch_x and batch_y to float type tensors
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                output = linear_model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                if step % 100 == 0 and loss.item() <= 0.5:
                    wandb.log({"train_loss": loss.item(),
                               "epoch": epoch,
                               "step": step})
                train_loss_list.append(loss.item())
            print("\t\tTrain loss: ", np.mean(train_loss_list))
            wandb.log({"train_loss_epoch": np.mean(train_loss_list),
                       "epoch": epoch})
                
            linear_model.eval()
            with torch.no_grad():
                test_loss_list = []
                for step, batch in enumerate(test_dataloader):
                    batch_x = batch[0]
                    batch_y = batch[1]
                    batch_x = batch_x.float()
                    batch_y = batch_y.float()
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    output = linear_model(batch_x)
                    loss = criterion(output, batch_y)
                    test_loss_list.append(loss.item())
                if np.mean(test_loss_list) <= 0.5:
                    wandb.log({"test_loss": np.mean(test_loss_list),
                           "epoch": epoch})
                
        wandb.finish()
    print("Finished training")
    
    

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)