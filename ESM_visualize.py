import argparse
import pathlib
import string
import sys, os
import torch
import torch.optim as optim
import pandas as pd
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import itertools
from typing import List, Tuple
import numpy as np
import models
import torch
import torch.nn as nn

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
    
    
    return parser

def esm_inference(df, esm_model, args, batch_converter, name):
    batch_size = args.batch_size
    sequence = df['sequence'].apply(remove_insertions).to_list()
    max_length = max([len(i) for i in sequence])
    target = df['target'].to_list()
    repre_l = []
    print(len(sequence))
    print(sequence[0])
    for i in tqdm(range(len(sequence))[::batch_size]):
        print(i)
        data = [("protein{}".format(j), sequence[j]) for j in range(len(sequence))[i:i+batch_size]]
        data_y = target[i:i+batch_size]
#        data_y = torch.tensor(data_y.to_numpy(), dtype=torch.long)
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        
        with torch.no_grad():
            results = esm_model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)
        repre = results['representations'][33]
        repre = repre.transpose(1,2)
        repre = repre.mean(dim=2)
        repre = repre.view(repre.shape[0], -1)
        repre_l.append(repre)
        # print(repre.shape)
    
    # Make the list of (8, 1280) tensors to one (8*n, 1280) tenosr, n is the length of the list
    repre = torch.cat(repre_l, dim=0)
    # Save the tensor to a file
    torch.save(repre, "{}_{}.pt".format(args.model_location, name))
        
        
def main(args):
    batch_size = args.batch_size
    
    df = pd.read_csv(args.data_input)
    NUDT15 = df[df['protein_type'] == 'NUDT15']
    PTEN = df[df['protein_type'] == 'PTEN'].reindex()
    CYP2C9 = df[df['protein_type'] == 'CYP2C9'].reindex()

    
    for model_location in args.model_location:
        esm_model, alphabet = pretrained.load_model_and_alphabet(model_location)
        esm_model.eval()
    esm_model, alphabet = pretrained.load_model_and_alphabet(model_location)

    if torch.cuda.is_available():
        esm_model = esm_model.cuda()
        print("Transferred model to GPU")
        
    batch_converter = alphabet.get_batch_converter()
    
#    esm_inference(NUDT15, esm_model, args, batch_converter, "NUDT15")
#    print("NUDT15 done")
    esm_inference(PTEN, esm_model, args, batch_converter, "PTEN")
    print("PTEN done")
    esm_inference(CYP2C9, esm_model, args, batch_converter, "CYP2C9")
    print("CYP2C9 done")
                
#                print("i", i, "\ni+8", i+8)
        
#                print(train_index[i:i+8])
        
#                print("data length", len(data))
#                print(data_y.to_numpy())
        
#                print(len(data), i, train_index[i])
        
                
        

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)