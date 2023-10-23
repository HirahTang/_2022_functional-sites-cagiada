import sys, os

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj as md
import random
import seaborn as sns
import re

import argparse

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score


from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool,cv
from catboost.datasets import titanic
from graphviz import Digraph

import warnings
warnings.filterwarnings('ignore')

#####
# Reproduce the feature generation process of the original dataset

alphabetAA_L_D={'-':0,'_' :0,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}
alphabetAA_D_L={v: k for k, v in alphabetAA_L_D.items()}
               
alphabetAA_3L_1L={"Ter":'*',"Gap":'-',"Ala":'A',"Cys":'C',"Asp":'D',"Glu":'E',"Phe":'F',"Gly":'G',"His":'H',"Ile":'I',"Lys":'K',"Leu":'L',"Met":'M',"Asn":'N',"Pro":'P',"Gln":'Q',"Arg":'R',"Ser":'S',"Thr":'T',"Val" :'V',"Trp":'W',"Tyr":'Y'}
alphabetAA_1L_3L={v: k for k,v in alphabetAA_3L_1L.items()}

alphabetAA_3L_1LC={"Ter":'*',"Gap":'-',"ALA":'A',"CYS":'C',"ASP":'D',"GLU":'E',"PHE":'F',"GLY":'G',"HIS":'H',"ILE":'I',"LYS":'K',"LEU":'L',"MET":'M',"ASN":'N',"PRO":'P',"GLN":'Q',"ARG":'R',"SER":'S',"THR":'T',"VAL" :'V',"TRP":'W',"TYR":'Y'}

alphabetAA_to_classes={'A':1,'C':2,'D':3,'E':3,'F':1,'G':4,'H':3,'I':1,'K':3,'L':1,'M':1,'N':2,'P':4,'Q':2,'R':3,'S':2,'T':2,'V':1,'W':1,'Y':1}
alphabetclasses_to_AA={v: k for k,v in alphabetAA_to_classes.items()}

AA_ddsp_number={'H' :0,'B' : 1,'E' : 2,'G' : 3,'I' : 4,'T' : 5,'S' : 6,' ' : 7}
AA_number_ddsp={v: k for k,v in AA_ddsp_number.items()}

AA_ddsp_number_simp={'H' :0,'E' : 1,'C' : 2}
AA_number_ddsp_simp={v: k for k,v in AA_ddsp_number.items()}

AA_to_hydrophobicity_scores={'A':44,'C':50,'D':-37,'E':-12,'F':96,'G':0,'H':-16,'I':100,'K':-30,'L':99,'M':74,'N':-35,'P':-46,'Q':-14,'R':-20,'S':-6,'T':13,'V':78,'W':90,'Y':57}

def remove_WT_score(score,WT_seq):
    for i in range(len(WT_seq)):
        score[i,alphabetAA_L_D[WT_seq[i]]-1]=np.nan
    return score

def load_data_V2(data,wt_seq,start_gap=0,column_score=1):
    df=pd.read_csv(data, delim_whitespace=True, comment='#')
    mutation_load=np.array(df.iloc[:,0])
    score_load=np.array(df.iloc[:,column_score])
    scores=np.empty((len(wt_seq),20),dtype=float)
    scores[:]=np.nan
    for i in range(len(mutation_load)):
        if  mutation_load[i][len(mutation_load[i])-1]!= '=' and mutation_load[i][len(mutation_load[i])-1]!= '*' :
            scores[int(mutation_load[i][1:len(mutation_load[i])-1])-1+start_gap, alphabetAA_L_D[mutation_load[i][len(mutation_load[i])-1]]-1]= float(score_load[i])
    return scores

def normalize_minmax(scores):
    normalized_scores=np.copy(scores)
    c_min_act=np.amin(scores[~np.isnan(scores)])
    c_max_act=np.amax(scores[~np.isnan(scores)])
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            normalized_scores[i,j]=(scores[i,j]-c_min_act)/(c_max_act-c_min_act)
                                  
    return normalized_scores


def features_classification(list_features_x,list_output_y,WT, args):
    
    X=[]
    Y=[]
    mapping_pos=[] 
    
    for i in range(len(WT)):
        for j in range(20):
            if j!=(alphabetAA_L_D[WT[i]]-1):
                temp_x=[]
                temp_y=[]
                cond=True

                for elem in list_features_x:
                    if elem.ndim==1:
                        if np.isnan(elem[i])==True and not args.nan:
                            
                            cond=False
                            
                    else:
                        if np.isnan(elem[i,j])==True and not args.nan:
                            cond=False    
                
                for elem in list_output_y:
                    if elem.ndim==1:
                        if np.isnan(elem[i])==True and not args.nan:
                            cond=False
                    else:
                        if np.isnan(elem[i,j])==True and not args.nan:
                            cond=False 

                if cond==True:
                    
                    for elem in list_features_x:
                        
                        if elem.ndim==1:
                            temp_x.append(elem[i])
                        else:

                            temp_x.append(elem[i,j])

                    for elem in list_output_y:
                        if elem.ndim==1:
                            temp_y.append(elem[i])
                        else:
                            temp_y.append(elem[i,j])
                    
                if len(temp_x)>0:
                    X.append(temp_x)
                    Y.append(temp_y)
                    mapping_pos.append([i,j])
#    print(X)
#    print(mapping_pos)
#    print(len(X))
#    print([len(w) for w in X])
#    print(len(Y))
#    print(Y)
    return np.array(X),Y,mapping_pos  

def create_parser():
    parser = argparse.ArgumentParser(
            description='Settings for generating data'
    )
    parser.add_argument(
            '--test', type=int,
            help='open the testing phase, AKA do not save the produced data',
            default=0,
    )
    
    parser.add_argument(
            '--nan', type=int,
            help='open the nan phase, AKA do not remove the nan values',
            default=0,
    )
    
    parser.add_argument(
            '--output_csv', type=str,
            help='output filepath for scores of variant sequences',
            default='output_adj.csv',
    )
    return parser
    
def multiclass_threshold(data_x,data_y,t_x,t_y):
    labels=np.copy(data_x)
    labels[:]=np.nan
    
    for i in range(data_x.shape[0]):
        for j in range(data_x.shape[1]):
            if np.isnan(data_x[i,j])!= True and np.isnan(data_y[i,j])!=True:
                if data_x[i,j] > t_x and data_y[i,j]<t_y:
                    labels[i,j]=1
                elif data_x[i,j] < t_x and data_y[i,j]<t_y:
                    labels[i,j]=2
                elif data_x[i,j] < t_x and data_y[i,j]>t_y:
                    labels[i,j]=3
                else:
                    labels[i,j]=0
    
    return labels      

def position_mean(score):
    score_mean=np.zeros(score.shape[0],dtype=float)
    for i in range(score.shape[0]):
        count=0
        flag_nan=True
        for j in range(score.shape[1]):
            if np.isnan(score[i,j])==False:
                flag_nan=False
                score_mean[i]+=score[i,j]
                count+=1
            else:
                pass
        if flag_nan==True:
            score_mean[i]=np.nan
        score_mean[i]/=count
        
    return score_mean

def normalize_cutoff(scores,lowcut,highcut):
    normalized_scores=np.copy(scores)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            if scores[i,j] < lowcut:
                normalized_scores[i,j]=lowcut
            elif scores[i,j] > highcut:
                normalized_scores[i,j]=highcut
            else:
                normalized_scores[i,j]=scores[i,j]
    return normalized_scores

def WCN(pdb_loc,scheme_e,WT):
    r0=7.0
    pdb=md.load(pdb_loc)
    topology=pdb.topology
    chainA=topology.select('chainid 0 and protein')
    pdb_chain0=pdb.atom_slice(chainA)
    pdb_dist,pdb_rp=md.compute_contacts(pdb_chain0,scheme=scheme_e,periodic=False)
    
    cm= md.geometry.squareform(pdb_dist,pdb_rp)[0]
    wcn=np.zeros((len(WT)),dtype=float)
    
    cm_adj=np.empty((len(WT),len(WT)),dtype=float)
    cm_adj[:]=np.nan
    chainA_top=pdb_chain0.topology
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i==0 and j==0:
                print(str(chainA_top.residue(i)))
            cm_adj[int(str(chainA_top.residue(i))[3:])-1,int(str(chainA_top.residue(j))[3:])-1]=cm[i,j]
    for i in range(len(WT)):
        nan_flag=True
        for j in range(len(WT)):
            if np.isnan(cm_adj[i,j])!=True and cm_adj[i,j]!=0.0:
                nan_flag=False
                wcn[i]+=(1-(cm_adj[i,j]*10/r0)**6)/(1-(cm_adj[i,j]*10/r0)**12)
        if nan_flag==True:
            wcn[i]=np.nan
    return wcn

def neighbor_scores(score,ext_range):
    score_neighborhood=np.zeros(len(score),dtype=float)
    for i in range(len(score)):
        if np.isnan(score[i])!=True:
            count_nan=0
            if i==0:
                for j in range(1,ext_range+1):
                    if np.isnan(score[j])==False:
                        score_neighborhood[i]+=score[j]
                    else:
                        count_nan+=1
                if count_nan!=ext_range:    
                    score_neighborhood[i]/=(ext_range)
                else:
                    score_neighborhood[i]=np.nan

            elif i==(len(score)-1):
                for j in range(len(score)-1-ext_range,len(score)-1):
                    if np.isnan(score[j])==False:
                        score_neighborhood[i]+=score[j]
                    else:
                        count_nan+=1
                if count_nan!=ext_range: 
                    score_neighborhood[i]/=ext_range
                else:
                    score_neighborhood[i]=np.nan                
            elif i<ext_range:
                for j in range(0,i+ext_range+1):
                    if j!=i:
                        if np.isnan(score[j])==False:
                            score_neighborhood[i]+=score[j]
                        else:
                            count_nan+=1
                if count_nan!=(i+ext_range):    
                    score_neighborhood[i]/=(i+ext_range)
                else:
                    score_neighborhood[i]=np.nan                        

            elif i>(len(score)-1-ext_range):
                for j in range(i-ext_range,len(score)):
                    if j!=i:
                        if np.isnan(score[j])==False:
                            score_neighborhood[i]+=score[j]
                        else:
                            count_nan+=1
                if count_nan!=(len(score)-i+ext_range):                     
                    score_neighborhood[i]/=(len(score)-i+ext_range)
                else:
                    score_neighborhood[i]=np.nan  
            else:
                for j in range(i-ext_range,i+ext_range+1):
                    if j!=i:
                        if np.isnan(score[j])==False:
                            score_neighborhood[i]+=score[j]
                        else:
                            count_nan+=1
                if count_nan!=(2*ext_range):  
                    score_neighborhood[i]/=(2*ext_range)
                else:
                    score_neighborhood[i]=np.nan             
        else:
            score_neighborhood[i]=np.nan
    return score_neighborhood

def label_loading(CYP2C9_WT_sequence, NUDT15_WT_sequence, PTEN_WT_sequence):
    # Load CYP2C9
    df=pd.read_csv('./score_maves/CYP2C9_data.csv',sep=',')
    mutation_load=np.array(df.iloc[:,0])
    score_funct=np.array(df.iloc[:,6])
    score_abund=np.array(df.iloc[:,17])
    
    CYP2C9_function=np.empty((len(CYP2C9_WT_sequence),20),dtype=float)
    CYP2C9_function[:]=np.nan  
    
    for i in range(len(mutation_load)):
        if score_funct[i] != 'NaN' and (mutation_load[i][-1:])!='~' and mutation_load[i][-1:]!='*' and mutation_load[i][-1:]!='X' and mutation_load[i] !="WT":
            CYP2C9_function[int(mutation_load[i][1:-1])-1, alphabetAA_L_D[mutation_load[i][-1:]]-1]=float(score_funct[i])
            
    CYP2C9_abundance=np.empty((len(CYP2C9_WT_sequence),20),dtype=float)
    CYP2C9_abundance[:]=np.nan  
    
    for i in range(len(mutation_load)):
        if score_abund[i] != 'NaN' and (mutation_load[i][-1:])!='~' and mutation_load[i][-1:]!='*' and mutation_load[i][-1:]!='X' and mutation_load[i] !="WT":
            CYP2C9_abundance[int(mutation_load[i][1:-1])-1, alphabetAA_L_D[mutation_load[i][-1:]]-1]=float(score_abund[i])
    
    CYP2C9_abundance_norm=normalize_minmax(CYP2C9_abundance)
    CYP2C9_function_norm=normalize_minmax(CYP2C9_function)
    
    # Target: CYP2C9_function, CYP2C9_abundance
    
    # Load NUDT15
    
    df=pd.read_excel('./score_maves/NUDT15_scores_pnas.xlsx', index_col=0,header=2)
    mutation_load=np.array(df.iloc[:,2])
    mutation_position=np.array(df.iloc[:,3])
    score_funct=np.array(df.iloc[:,7])
    score_abund=np.array(df.iloc[:,5])

    NUDT15_function=np.empty((len(NUDT15_WT_sequence),20),dtype=float)
    NUDT15_function[:]=np.nan
    
    for i in range(len(mutation_load)):
        if score_funct[i] != ' NA':
            NUDT15_function[int(mutation_position[i])-1, alphabetAA_L_D[mutation_load[i][len(mutation_load[i])-1]]-1]=float(score_funct[i])
    
    NUDT15_function_norm=normalize_minmax(NUDT15_function)
    
    NUDT15_abundance=np.empty((len(NUDT15_WT_sequence),20),dtype=float)
    NUDT15_abundance[:]=np.nan
    
    for i in range(len(mutation_load)):
        if score_abund[i] != ' NA':
            NUDT15_abundance[int(mutation_position[i])-1, alphabetAA_L_D[mutation_load[i][len(mutation_load[i])-1]]-1]=float(score_abund[i])
    
    NUDT15_abundance_norm=normalize_minmax(NUDT15_abundance)
    
    # Target: NUDT15_function, NUDT15_abundance
    
    # Load PTEN
    
    df = pd.read_excel (r'./score_maves/PTEN_MAVE_act.xlsx', sheet_name='scores',header=1)
    mutation_load=np.array(df.iloc[:,0])
    score_funct=np.array(df.iloc[:,6])
    
    PTEN_function=np.empty((len(PTEN_WT_sequence),20),dtype=float)
    PTEN_function[:]=np.nan
    
    for i in range(len(mutation_load)):
        if score_funct[i] != 'NaN' and (mutation_load[i][-1:])!='~' and mutation_load[i][-1:]!='*':
            PTEN_function[int(mutation_load[i][1:-1])-1, alphabetAA_L_D[mutation_load[i][-1:]]-1]=float(score_funct[i])
            
    PTEN_function_norm=normalize_minmax(PTEN_function)
    
    df=pd.read_csv('./score_maves/PTEN_VAMP_seq.csv',sep=',')
    mutation_load=np.array(df.iloc[:,2],dtype=str)
    score_abund=np.array(df.iloc[:,3])
    
    PTEN_abundance=np.empty((len(PTEN_WT_sequence),20),dtype=float) 
    PTEN_abundance[:]=np.nan
    
    for i in range(len(mutation_load)):
        if score_abund[i] != 'NaN' and mutation_load[i][-1:]!='=' and alphabetAA_3L_1L[mutation_load[i][-3:]]!='*':  
            PTEN_abundance[int(mutation_load[i][5:-3])-1, alphabetAA_L_D[alphabetAA_3L_1L[mutation_load[i][-3:]]]-1]=float(score_abund[i])
            
    PTEN_abundance_norm=normalize_minmax(PTEN_abundance)
    
    # Target: PTEN_function, PTEN_abundance
    NUDT15_labels=multiclass_threshold(NUDT15_abundance,NUDT15_function,0.38,0.51)
    PTEN_labels=multiclass_threshold(PTEN_abundance,PTEN_function,0.54,-0.9)
    CYP2C9_labels=multiclass_threshold(CYP2C9_abundance,CYP2C9_function,0.38,0.40)
    
    return [NUDT15_labels, NUDT15_abundance_norm, NUDT15_function_norm], \
        [PTEN_labels, PTEN_abundance_norm, PTEN_function_norm], \
            [CYP2C9_labels, CYP2C9_abundance_norm, CYP2C9_function_norm]

def computational_x(CYP2C9_WT_sequence, NUDT15_WT_sequence, PTEN_WT_sequence):
    
    #NUDT15
    
    NUDT15_GEMME=load_data_V2('./scores_GEMME/prism_gemme_Q9NV35.txt',NUDT15_WT_sequence)
    NUDT15_GEMME=remove_WT_score(NUDT15_GEMME,NUDT15_WT_sequence)
    
    NUDT15_GEMME_mean=position_mean(NUDT15_GEMME)

    NUDT15_rosetta_ddg=load_data_V2('./scores_rosetta/prism_rosetta_ddg_Q9NV35.txt',NUDT15_WT_sequence)
    NUDT15_rosetta_ddg=NUDT15_rosetta_ddg/2.9
    NUDT15_rosetta_ddg=remove_WT_score(NUDT15_rosetta_ddg,NUDT15_WT_sequence)
    
    NUDT15_rosetta_ddg_norm=normalize_cutoff(NUDT15_rosetta_ddg,0.0,5.0)
    
    NUDT15_rosetta_ddg_mean=position_mean(NUDT15_rosetta_ddg_norm)
    
    #PTEN
    
    PTEN_GEMME=load_data_V2('./scores_GEMME/prism_gemme_P60484.txt',PTEN_WT_sequence)
    PTEN_GEMME=remove_WT_score(PTEN_GEMME,PTEN_WT_sequence)
    
    PTEN_GEMME_mean=position_mean(PTEN_GEMME)
    
    PTEN_rosetta_ddg=load_data_V2('./scores_rosetta/prism_rosetta_ddg_P60484.txt',PTEN_WT_sequence)
    PTEN_rosetta_ddg=PTEN_rosetta_ddg
    PTEN_rosetta_ddg=remove_WT_score(PTEN_rosetta_ddg,PTEN_WT_sequence)
    
    PTEN_rosetta_ddg_norm=normalize_cutoff(PTEN_rosetta_ddg,0.0,5.0)
    
    PTEN_rosetta_ddg_mean=position_mean(PTEN_rosetta_ddg_norm)
    
    #CYP2C9
    
    CYP2C9_GEMME=load_data_V2('./scores_GEMME/prism_gemme_P11712.txt',CYP2C9_WT_sequence)
    CYP2C9_GEMME=remove_WT_score(CYP2C9_GEMME,CYP2C9_WT_sequence)
    
    CYP2C9_GEMME_mean=position_mean(CYP2C9_GEMME)
    
    CYP2C9_rosetta_ddg=load_data_V2("./scores_rosetta/prism_rosetta_ddg_P11712.txt",CYP2C9_WT_sequence,29) # CYP2C9_WT_seq_rose
    CYP2C9_rosetta_ddg=remove_WT_score(CYP2C9_rosetta_ddg,CYP2C9_WT_sequence) # CYP2C9_WT_seq_rose
    
    CYP2C9_rosetta_ddg_norm=normalize_cutoff(CYP2C9_rosetta_ddg,0,5)
    
    CYP2C9_rosetta_ddg_mean=position_mean(CYP2C9_rosetta_ddg_norm)
    
    ## Hydrophobicity
    
    # NUDT15
    
    NUDT15_hydrophobicity_mut=np.empty((len(NUDT15_WT_sequence),20),dtype=float)
    NUDT15_hydrophobicity_mut[:]=np.nan

    for i in range(len(NUDT15_WT_sequence)):
        for j in range(20):
            NUDT15_hydrophobicity_mut[i,j]=AA_to_hydrophobicity_scores[alphabetAA_D_L[j+1]]
            
    # PTEN
    
    PTEN_hydrophobicity_mut=np.empty((len(PTEN_WT_sequence),20),dtype=float)
    PTEN_hydrophobicity_mut[:]=np.nan
    
    for i in range(len(PTEN_WT_sequence)):
        for j in range(20):
            PTEN_hydrophobicity_mut[i,j]=AA_to_hydrophobicity_scores[alphabetAA_D_L[j+1]]
    
    # CYP2C9
    
    CYP2C9_hydrophobicity_mut=np.empty((len(CYP2C9_WT_sequence),20),dtype=float)
    CYP2C9_hydrophobicity_mut[:]=np.nan
    
    for i in range(len(CYP2C9_WT_sequence)):
        for j in range(20):
            CYP2C9_hydrophobicity_mut[i,j]=AA_to_hydrophobicity_scores[alphabetAA_D_L[j+1]]
    
    
    ## WCN
    
    # NUDT15
    
    NUDT15_wcn=WCN('./pdbs/Q9NV35_5lpg.pdb','ca',NUDT15_WT_sequence)
    
    PTEN_wcn=WCN('./pdbs/P60484_1d5r.pdb','ca',PTEN_WT_sequence)
    
    CYP2C9_wcn=WCN('./pdbs/P11712_1og5.pdb','ca',CYP2C9_WT_sequence)
    
    ## Neighbors sequences scores
    
    NUDT15_rosetta_neigbor_scores=neighbor_scores(NUDT15_rosetta_ddg_mean,1)

    NUDT15_GEMME_neigbor_scores=neighbor_scores(NUDT15_GEMME_mean,1)

    PTEN_rosetta_neigbor_scores=neighbor_scores(PTEN_rosetta_ddg_mean,1)

    PTEN_GEMME_neigbor_scores=neighbor_scores(PTEN_GEMME_mean,1)

    CYP2C9_rosetta_neigbor_scores=neighbor_scores(CYP2C9_rosetta_ddg_mean,1)

    CYP2C9_GEMME_neigbor_scores=neighbor_scores(CYP2C9_GEMME_mean,1)
    return [NUDT15_GEMME, NUDT15_rosetta_ddg_norm,NUDT15_GEMME_mean,NUDT15_rosetta_ddg_mean,NUDT15_hydrophobicity_mut,NUDT15_GEMME_neigbor_scores,NUDT15_rosetta_neigbor_scores,NUDT15_wcn], \
            [PTEN_GEMME, PTEN_rosetta_ddg_norm,PTEN_GEMME_mean,PTEN_rosetta_ddg_mean,PTEN_hydrophobicity_mut,PTEN_GEMME_neigbor_scores,PTEN_rosetta_neigbor_scores,PTEN_wcn], \
                [CYP2C9_GEMME, CYP2C9_rosetta_ddg_norm,CYP2C9_GEMME_mean,CYP2C9_rosetta_ddg_mean,CYP2C9_hydrophobicity_mut,CYP2C9_GEMME_neigbor_scores,CYP2C9_rosetta_neigbor_scores,CYP2C9_wcn]
    
def main(args):
    CYP2C9_WT_sequence="MDSLVVLVLCLSCLLLLSLWRQSSGRGKLPPGPTPLPVIGNILQIGIKDISKSLTNLSKVYGPVFTLYFGLKPIVVLHGYEAVKEALIDLGEEFSGRGIFPLAERANRGFGIVFSNGKKWKEIRRFSLMTLRNFGMGKRSIEDRVQEEARCLVEELRKTKASPCDPTFILGCAPCNVICSIIFHKRFDYKDQQFLNLMEKLNENIKILSSPWIQICNNFSPIIDYFPGTHNKLLKNVAFMKSYILEKVKEHQESMDMNNPQDFIDCFLMKMEKEKHNQPSEFTIESLENTAVDLFGAGTETTSTTLRYALLLLLKHPEVTAKVQEEIERVIGRNRSPCMQDRSHMPYTDAVVHEVQRYIDLLPTSLPHAVTCDIKFRNYLIPKGTTILISLTSVLHDNKEFPNPEMFDPHHFLDEGGNFKKSKYFMPFSAGKRICVGEALAGMELFLFLTSILQNFNLKSLVDPKNLDTTPVVNGFASVPPFYQLCFIPV"
    CYP2C9_WT_seq_rose="-----------------------------PPGPTPLPVIGNILQIGIKDISKSLTNLSKVYGPVFTLYFGLKPIVVLHGYEAVKEALIDLGEEFSGRGIFPLAERANRGFGIVFSNGKKWKEIRRFSLMTLRNFGMGKRSIEDRVQEEARCLVEELRKTKASPCDPTFILGCAPCNVICSIIFHKRFDYKDQQFLNLMEKLNENIEILSSPWIQVYNNFPALLDYFPGTHNKLLKNVAFMKSYILEKVKEHQESMDMNNPQDFIDCFLMKMEKEKHNQPSEFTIESLENTAVDLFGAGTETTSTTLRYALLLLLKHPEVTAKVQEEIERVIGRNRSPCMQDRSHMPYTDAVVHEVQRYIDLLPTSLPHAVTCDIKFRNYLIPKGTTILISLTSVLHDNKEFPNPEMFDPHHFLDEGGNFKKSKYFMPFSAGKRICVGEALAGMELFLFLTSILQNFNLKSLVDPKNLDTTPVVNGFASVPPFYQLCFIPV"
    NUDT15_WT_sequence="MTASAQPRGRRPGVGVGVVVTSCKHPRCVLLGKRKGSVGAGSFQLPGGHLEFGETWEECAQRETWEEAALHLKNVHFASVVNSFIEKENYHYVTILMKGEVDVTHDSEPKNVEPEKNESWEWVPWEELPPLDQLFWGLRCLKEQGYDPFKEDLNHLVGYKGNHL"
    PTEN_WT_sequence="MTAIIKEIVSRNKRRYQEDGFDLDLTYIYPNIIAMGFPAERLEGVYRNNIDDVVRFLDSKHKNHYKIYNLCAERHYDTAKFNCRVAQYPFEDHNPPQLELIKPFCEDLDQWLSEDDNHVAAIHCKAGKGRTGVMICAYLLHRGKFLKAQEALDFYGEVRTRDKKGVTIPSQRRYVYYYSYLLKNHLDYRPVALLFHKMMFETIPMFSGGTCNPQFVVCQLKVKIYSSNSGPTRREDKFMYFEFPQPLPVCGDIKVEFFHKQNKMLKKDKMFHFWVNTFFIPGPEETSEKVENGSLCDQEIDSICSIERADNDKEYLVLTLTKNDLDKANKDKANRYFSPNFKVKLYFTKTVEEPSNPEASSSTSVTPDVSDNEPDHYRYSDTTDSDPENEPFDEDQHTQITKV"

    
    NUDT15_Xs, PTEN_Xs, CYP2C9_Xs = computational_x(CYP2C9_WT_sequence, NUDT15_WT_sequence, PTEN_WT_sequence)
    
    NUDT15_all, PTEN_all, CYP2C9_alls=label_loading(CYP2C9_WT_sequence, NUDT15_WT_sequence, PTEN_WT_sequence)
    
    NUDT15_labels, NUDT15_abundance_norm, NUDT15_function_norm = NUDT15_all[0], NUDT15_all[1], NUDT15_all[2]
    PTEN_labels, PTEN_abundance_norm, PTEN_function_norm = PTEN_all[0], PTEN_all[1], PTEN_all[2]
    CYP2C9_labels, CYP2C9_abundance_norm, CYP2C9_function_norm = CYP2C9_alls[0], CYP2C9_alls[1], CYP2C9_alls[2]
    
    
    print(NUDT15_labels.shape, PTEN_labels.shape, CYP2C9_labels.shape)
    
#    NUDT15_X = [(164, 20), (164, 20), (164,), (164,), (164, 20), (164,), (164,), (164,)]
#    PTEN_X = [(403, 20), (403, 20), (403,), (403,), (403, 20), (403,), (403,), (403,)]
#    CYP2C9_X = [(490, 20), (490, 20), (490,), (490,), (490, 20), (490,), (490,), (490,)]
#    NUDT15_X,NUDT15_Y,NUDT15_map=features_classification([np.random.rand(*x) for x in NUDT15_X],[NUDT15_labels],NUDT15_WT_sequence)
#    PTEN_X,PTEN_Y,PTEN_map=features_classification([np.random.rand(*x) for x in PTEN_X],[PTEN_labels],PTEN_WT_sequence)
#    CYP2C9_X,CYP2C9_Y,CYP2C9_map=features_classification([np.random.rand(*x) for x in CYP2C9_X],[CYP2C9_labels],CYP2C9_WT_sequence)

#    print("The output of importance:\n", NUDT15_labels.shape, NUDT15_abundance_norm.shape, NUDT15_function_norm.shape)
#    print(NUDT15_abundance_norm, PTEN_abundance_norm, CYP2C9_abundance_norm)

#    print(PTEN_Xs)
#    print(CYP2C9_Xs)
    
    _, NUDT15_abundance_norm_Y, _ = features_classification(NUDT15_Xs, [NUDT15_abundance_norm], NUDT15_WT_sequence, args)
    _, NUDT15_function_norm_Y, _ = features_classification(NUDT15_Xs, [NUDT15_function_norm], NUDT15_WT_sequence, args)
    
    _, PTEN_abundance_norm_Y, _ = features_classification(PTEN_Xs, [PTEN_abundance_norm], PTEN_WT_sequence, args)
    _, PTEN_function_norm_Y, _ = features_classification(PTEN_Xs, [PTEN_function_norm], PTEN_WT_sequence, args)
    
    _, CYP2C9_abundance_norm_Y, _ = features_classification(CYP2C9_Xs, [CYP2C9_abundance_norm], CYP2C9_WT_sequence, args)
    _, CYP2C9_function_norm_Y, _ = features_classification(CYP2C9_Xs, [CYP2C9_function_norm], CYP2C9_WT_sequence, args)

    NUDT15_X,NUDT15_Y,NUDT15_map=features_classification(NUDT15_Xs,[NUDT15_labels],NUDT15_WT_sequence, args)
    PTEN_X,PTEN_Y,PTEN_map=features_classification(PTEN_Xs,[PTEN_labels],PTEN_WT_sequence, args)
    CYP2C9_X, CYP2C9_Y, CYP2C9_map=features_classification(CYP2C9_Xs,[CYP2C9_labels],CYP2C9_WT_sequence, args)
    
    
    
    print("Finished")
    print("NUDT15: ",len(NUDT15_Y), len(NUDT15_map))
    print("PTEN: ",len(PTEN_Y), len(PTEN_map))
    print("CYP2C9: ",len(CYP2C9_Y), len(CYP2C9_map))
    print(NUDT15_X.shape, PTEN_X.shape, CYP2C9_X.shape)
    print(NUDT15_map[:20])
    
    print("Norm test:\n")
    print(len(NUDT15_abundance_norm_Y), len(NUDT15_function_norm_Y))
    print(len(PTEN_abundance_norm_Y), len(PTEN_function_norm_Y))
    print(len(CYP2C9_abundance_norm_Y), len(CYP2C9_function_norm_Y))
    
    for mut in NUDT15_map[:20]:
        mut_seq = NUDT15_WT_sequence
        mut_seq = mut_seq[:mut[0]] + alphabetAA_D_L[mut[1]] + mut_seq[mut[0]+1:] # Retrieve the whole mutated sequence
        # mut_seq[mut[0]] = alphabetAA_D_L[mut[1]]
        
        print("WT:", NUDT15_WT_sequence, '\n')
        print("Mutatant:", mut_seq, '\n')
        print("\n")
    mut_location = []
    aa_orig = []
    aa_mut = []    
    list_seq = []
    target = []
    protein_type = []
    
    for mut in NUDT15_map:
        mut_seq = NUDT15_WT_sequence
        mut_seq = mut_seq[:mut[0]] + alphabetAA_D_L[mut[1]] + mut_seq[mut[0]+1:]
        mut_location.append(mut[0])
        aa_orig.append(NUDT15_WT_sequence[mut[0]])
        aa_mut.append(alphabetAA_D_L[mut[1]])
        list_seq.append(mut_seq)
        protein_type.append("NUDT15")
    
    for mut in PTEN_map:
        mut_seq = PTEN_WT_sequence
        mut_seq = mut_seq[:mut[0]] + alphabetAA_D_L[mut[1]] + mut_seq[mut[0]+1:]
        mut_location.append(mut[0])
        aa_orig.append(PTEN_WT_sequence[mut[0]])
        aa_mut.append(alphabetAA_D_L[mut[1]])
        list_seq.append(mut_seq)
        protein_type.append("PTEN")
        
    for mut in CYP2C9_map:
        mut_seq = CYP2C9_WT_sequence
        mut_seq = mut_seq[:mut[0]] + alphabetAA_D_L[mut[1]] + mut_seq[mut[0]+1:]
        mut_location.append(mut[0])
        aa_orig.append(CYP2C9_WT_sequence[mut[0]])
        aa_mut.append(alphabetAA_D_L[mut[1]])
        list_seq.append(mut_seq)
        protein_type.append("CYP2C9")
        
    output_df = pd.DataFrame(np.concatenate((NUDT15_X, PTEN_X, CYP2C9_X), axis=0))
    abundance_norm = np.concatenate((NUDT15_abundance_norm_Y, PTEN_abundance_norm_Y, CYP2C9_abundance_norm_Y), axis=0)
    function_norm = np.concatenate((NUDT15_function_norm_Y, PTEN_function_norm_Y, CYP2C9_function_norm_Y), axis=0)
    target = np.concatenate((NUDT15_Y, PTEN_Y, CYP2C9_Y), axis=0)
    output_df['target'] = target
    output_df['mutation_location'] = mut_location
    output_df['aa_orig'] = aa_orig
    output_df['mutate_to'] = aa_mut
    output_df['sequence'] = list_seq
    output_df['protein_type'] = protein_type
    output_df['abundance_score'] = abundance_norm
    output_df['function_score'] = function_norm
    if not args.test:
        output_df.to_csv(args.output_csv, index=False)
        

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
    