import argparse
from biotite.sequence.io.fasta import FastaFile, get_sequences
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import esm
import esm.inverse_folding

# Calculate the fitness scores by ESM-IF

def score_singlechain_backbone(model, alphabet, args, protein_df):
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
    coords, native_seq = esm.inverse_folding.util.load_coords(args.pdbfile, args.chain)
    print('Native sequence loaded from structure file:')
    print(native_seq)
    print('\n')

    ll, _ = esm.inverse_folding.util.score_sequence(
            model, alphabet, coords, native_seq) 
    print('Native sequence')
    print(f'Log likelihood: {ll:.2f}')
    print(f'Perplexity: {np.exp(-ll):.2f}')

    print('\nScoring variant sequences from sequence file..\n')
    
    """
    if args.seqfile.endswith('.fasta'):
        infile = FastaFile()
        infile.read(args.seqfile)
        seqs = get_sequences(infile)
        Path(args.outpath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(args.outpath, 'w') as fout:
            fout.write('seqid,log_likelihood\n')
            for header, seq in tqdm(seqs.items()):
                ll, _ = esm.inverse_folding.util.score_sequence(
                        model, alphabet, coords, str(seq))
                fout.write(header + ',' + str(ll) + '\n')
        print(f'Results saved to {args.outpath}') 
    else:
        Path(args.outpath).parent.mkdir(parents=True, exist_ok=True)
        seq = args.seqfile
        with open(args.outpath, 'w') as fout:
            fout.write('seqid,log_likelihood\n')
            print(native_seq, len(native_seq))
            print(str(seq), len(str(seq)))
            ll, _ = esm.inverse_folding.util.score_sequence(
                        model, alphabet, coords, str(seq))
            fout.write("Seq" + ',' + str(ll) + '\n')
        print(f'Results saved to {args.outpath}') 
    """
    sub_df = protein_df[(protein_df['protein_type'] == args.protein_name) & ((protein_df['abundance_score'].isna()==False)|(protein_df['function_score'].isna()==False))]
    print(sub_df.shape)
    # reindex sub_df
    sub_df = sub_df.reset_index(drop=True)
    ESMIF_list = []
    
    for index, row in tqdm(sub_df.iterrows()):
        if args.protein_name == 'NUDT15':
            seq = row['sequence'][8:]
        elif args.protein_name == 'PTEN':
            PTEN_index = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 288, 290, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350]
            seq = ''
            for i in range(len(row['sequence'])):
                if i in PTEN_index:
                    seq+=row['sequence'][i]
        elif args.protein_name == 'CYP2C9':
            seq = row['sequence'][29:]
        else:
            print("Please choose from the three proteins: NUDT15, PTEN, CYP2C9")
            break
        seq = seq.replace('X', '')
        seq = seq.replace('_', '')
#        print(len(seq), len(native_seq))
        if len(seq) != len(native_seq):
            ESMIF_list.append(0)
            continue
        
        ll, _ = esm.inverse_folding.util.score_sequence(
                            model, alphabet, coords, str(seq))
#        print(ll)
        ESMIF_list.append(ll)


    print(len(ESMIF_list))
    print(len(sub_df.loc[:, 'ESMIF_score']))
    sub_df.loc[:, 'ESMIF_score'] = ESMIF_list
    sub_df.to_csv(args.outpath)
    
def create_parser():
    parser = argparse.ArgumentParser(
            description='Score sequences based on a given structure.'
    )
    parser.add_argument(
            '--pdbfile', type=str,
            help='input filepath, either .pdb or .cif',
    )
    parser.add_argument(   
            '--seqdata', type=str, 
            help='the location for the original datafile',
    )
    parser.add_argument(
            '--seqfile', type=str,
            help='input filepath for variant sequences in a .fasta file',
    )
    parser.add_argument(
            '--outpath', type=str,
            help='output filepath for scores of variant sequences',
            default='output_adj.csv',
    )
    parser.add_argument(
            '--chain', type=str,
            help='chain id for the chain of interest', default='A',
    )
    parser.set_defaults(multichain_backbone=False)
    parser.add_argument(
            '--multichain-backbone', action='store_true',
            help='use the backbones of all chains in the input for conditioning'
    )
    parser.add_argument(
            '--singlechain-backbone', dest='multichain_backbone',
            action='store_false',
            help='use the backbone of only target chain in the input for conditioning'
    )
    
    parser.add_argument(
            '--protein_name', type=str,
            help='Protein Name: Among three choices: NUDDT15, PTEN, CYP2C9', default='NUDT15',
    )
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    
    return parser
    

def main(args):
    
    
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    protein_df = pd.read_csv(args.seqdata, index_col=0)
    
    
    score_singlechain_backbone(model, alphabet, args, protein_df)



if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)