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

# Original Codes for the NoteBook
def wandb_init():
    wandb.init(project="ESM-1v-Functional-site", entity="hantang")
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
        "--pooling_method",
        type=str,
        default="mean",
        help="Pooling method",
    )
    parser.add_argument(
        "--fingerprints",
        type=int,
        default=1,
        help="1 for keep the ESM model unchanged, 0 for fine-tuning the ESM model",
    )
    
    parser.add_argument(
        "--save_model",
        type=int,
        default=0,
        help="1 for saving the model, 0 for not saving the model",
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
    
def eval_metrics(prediction, target):
    f1 = f1_score(target, prediction, average='macro')
    precision = precision_score(target, prediction, average='macro')
    recall = recall_score(target, prediction, average='macro')
    mcc = np.sqrt(abs(precision*recall))
    return mcc, f1, precision, recall

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
    df = df[(df['abundance_score'].isna()==False)&(df['function_score'].isna()==False)]
    df.reset_index(drop=True, inplace=True)
    print(df.shape)
    
    sequence = df['sequence'].apply(remove_insertions)
    max_length = max([len(i) for i in sequence])
    if args.target_type == "classification":
        target = df['target'].to_numopy()
    elif args.target_type == "regression":
        target = df[['abundance_score', 'function_score']].to_numpy()
    
    for model_location in args.model_location:
        esm_model, alphabet = pretrained.load_model_and_alphabet(model_location)
        
#    esm_model, alphabet = pretrained.load_model_and_alphabet(model_location)
#    if args.fingerprint:
#        esm_model.eval()
#    else:
#        esm_model.train()
    if torch.cuda.is_available():
        esm_model = esm_model.cuda()
        print("Transferred model to GPU")
        
    batch_converter = alphabet.get_batch_converter()
    
    
    
#    pooling = nn.AvgPool1d(166, stride=1).cuda()
    split_time = 0
    for train_index, test_index in split_method.split(sequence, target):
        if args.pooling_method == "aa":
            linear_model = models.ESMAttention1d(max_length=max_length, d_embedding=1280, target_class=target_class).cuda()
        elif args.pooling_method == "mean":
            linear_model = models.ESMAttention1dMean(d_embedding=1280, target_class=target_class).cuda()
        else:
            raise ValueError("Pooling method not supported")
        # Model Reinitialization
        if args.fingerprints:
            optimizer = optim.Adam(linear_model.parameters(), lr=args.lr)
        else:
            optimizer = optim.Adam(list(esm_model.parameters())+list(linear_model.parameters()), lr=args.lr)
        
        ###
        print("Start training")
        print("batch_size:", args.batch_size)
        print("pooling_method:", args.pooling_method)
        print("optimizer:", optimizer)
        
        ###
        print("Split:", split_time, '\n')
        split_time += 1
        
        
#        if split_time > 1: # Debug
#            break
        
        X_train, X_test = sequence[train_index], sequence[test_index]
        Y_train, Y_test = target[train_index], target[test_index]
        
#        print(train_index, test_index)
#        print(X_train.shape, X_test.shape)
        
        # WanDB Init
        wandb.init(
            project="ESM-1v-Functional-site",
            name="{}_{}_finetuned_{}_fingerprint".format(model_location, args.pooling_method, split_time, args.fingerprints),
            entity="hirahtang",
            config={
                "model_location": model_location,
                "data_input": args.data_input,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "pooling_method": args.pooling_method,
                "fingerprints": args.fingerprints,
                "save_model": args.save_model,
                "target_type": args.target_type,
                "split_time": split_time
            }
            )
        # ----------------
        
        for epoch in range(args.epochs):
            print("\tEpoch:", epoch)
            args.print_res = True
            linear_model.train()
            iterate_loss = []
            for i in tqdm(range(len(train_index))[::batch_size]):  
                
#                print("i", i, "\ni+8", i+8)
                data = [("protein{}".format(j), sequence[j]) for j in train_index[i:i+batch_size]]
#                print(train_index[i:i+8])
                data_y = target[train_index[i:i+batch_size]]

                data_y = torch.tensor(data_y, dtype=torch.float)

                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                
                if args.fingerprints:
                    esm_model.eval()
                    with torch.no_grad():
                        results = esm_model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)
                else:
                    esm_model.train()
                    results = esm_model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)
                repre = results['representations'][33]
                
#                print("Experiment starts here")
                
                
                if args.pooling_method == "mean":
                    repre = repre.transpose(1,2)
                    repre = repre.mean(dim=2)
                    repre = repre.view(repre.shape[0], -1)
#                print(repre.shape)
#                
#                print(repre.shape)
                if i > 10:
                    args.print_res = False
                train_loss = train_esm(linear_model, repre, data_y, criterion, optimizer, args)
                if i % 100 < batch_size:
                    wandb.log({"step": i,
                               "train_loss": train_loss,
                               "epoch": epoch})
                # loss = criterion(logits, data_y.cuda())
                iterate_loss.append(train_loss)
            wandb.log({"train_loss_epoch": sum(iterate_loss)/len(iterate_loss),
                       "epoch": epoch})
            print("\t\tTrain loss:", sum(iterate_loss)/len(iterate_loss))

#                break
            with torch.no_grad():
                args.print_res = True
                prediction_l = []
                target_l  =[]
                avg_loss = []
                linear_model.eval()
                for i in tqdm(range(len(test_index))[::batch_size]):  
                    test_data = [("protein{}".format(j), sequence[j]) for j in test_index[i:i+batch_size]]
                    # test_data = [("protein{}".format(i), sequence[i]),]
                    # data_y = target[train_index[i:i+8]]
                    test_y = target[test_index[i:i+batch_size]]
                    
                    # print(test_y)
                    test_y = torch.tensor(test_y, dtype=torch.float)
                    
                    # test_y = torch.tensor(data_y, dtype=torch.long)
#                    print(test_y)
                    batch_labels, batch_strs, batch_tokens = batch_converter(test_data)
                    if args.fingerprints:
                        esm_model.eval()
                    results = esm_model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)
                    test_repre = results['representations'][33]
                   
                    if args.pooling_method == "mean":
                        test_repre = test_repre.transpose(1,2)
                        test_repre = test_repre.mean(dim=2)
                        test_repre = test_repre.view(test_repre.shape[0], -1)
                        
                    
#                    print("Test Tensor shape", test_repre.shape)
                    if i >= 10:
                        args.print_res = False
                        
                    valid_loss, prediction, test_value = eval_esm(linear_model, test_repre, test_y, criterion, args)
                    avg_loss.append(valid_loss)
                    prediction_l.extend(prediction)
                    target_l.extend(test_value)
#                    print(avg_loss, len(prediction_l), len(target_l))
#                    if i > 40:
#                        break
                if args.target_type == "classification":
                    
                    eval_scores = eval_metrics(prediction_l, target_l)
                    
                    print("\t\tMCC:", eval_scores[0])
                    print("\t\tF1:", eval_scores[1])
                    print("\t\tPrecision:", eval_scores[2])
                    print("\t\tRecall:", eval_scores[3])
                    print("\t\tAverage loss:", sum(avg_loss)/len(avg_loss))
                    
                    wandb.log({"test_loss": sum(avg_loss)/len(avg_loss),
                                "epoch": epoch,
                            "MCC": eval_scores[0],
                            "F1": eval_scores[1],
                            "Precision": eval_scores[2],
                            "Recall": eval_scores[3]})
                elif args.target_type == "regression":
                    wandb.log({"test_loss": sum(avg_loss)/len(avg_loss),
                                "epoch": epoch})
                
                else:
                    raise ValueError("Target type not supported")
        if args.save_model and not args.fingerprints:
            torch.save(esm_model.state_dict(), "{}_{}_finetuned_{}.pt".format(model_location, args.pooling_method, split_time))
            print("Model saved to {}_{}_finetuned_{}.pt".format(model_location, args.pooling_method, split_time))           
        wandb.finish()
        
    print("End")
    
if __name__ == "__main__":
    wandb.login()
    parser = create_parser()
    args = parser.parse_args()
    print("code running")
    main(args)