import pandas as pd
import numpy as np
import random
import torch
import pickle
import math
from scipy.special import kl_div
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
import re

def read_fasta(file_path):
    '''
    Reads a FASTA file and returns a list of dictionaries containing header and sequence
    '''
    with open(file_path, "r") as file:
        fasta_data = file.read().strip().split(">")[1:]  # Skip the first empty split
    records = []
    for entry in fasta_data:
        lines = entry.split("\n")
        header = lines[0]
        sequence = "".join(lines[1:])  # Combine sequence lines
        records.append({"header": header, "sequence": sequence})
    return records

def tail_overlap(seq1,seq2):
    '''
    computes for how many basepairs there is sequence overlap
    '''
    for i in range(min(len(seq1),len(seq2))):
        if not (seq1[i]==seq2[i]):
            return i         
    return i


def score_similarities(pr_1,pr_2):
    '''
    returns the identity between two sequences of equal length
    '''
    assert len(pr_1)==len(pr_2)
    counter=0
    for i in range(len(pr_1)):
        if pr_1[i]==pr_2[i]:
            counter+=1

    return counter/len(pr_1)

def print_results(df):
    '''
    prints result overview from the dataframe analysis
    '''
    for i in df.columns:
        if df[i].dtype == bool:
            print(i,":",df[i].to_list().count(True),"/",len(df),"=",
                                     df[i].to_list().count(True)/len(df))

def convert_data(col_h, col_t, col_s, ht_i, s_i, df_s):
    '''
    restructures data to be suitable for the analysis
    order: [[real_FW,real_RC], [con_FW, con_RC],
    [head]
    '''
    
    collection={}

    intersections=set(df_s.Intersection)
    
    for i in tqdm(intersections, desc="Processing tasks", unit="task"):
        for hh in ht_i.keys():
            if hh[0]==i:
                h=hh[1]
                for tt in ht_i.keys():
                    if (tt[0]==i) and (tt[1]!=h):
                        t=tt[1]
                        collection[(i,h,t)]=[
                            [np.array(col_s[s_i[(i,h,h)]][0]),np.array(col_s[s_i[(i,h,h)]][1])],
                            [np.array(col_s[s_i[(i,h,t)]][0]),np.array(col_s[s_i[(i,h,t)]][1])],
                            [np.array(col_h[ht_i[(i,h)]])]
                        ]
                        
    return collection
    

def get_tail_info(collection_keys, tdict,mbp=800, bp=200):
    '''
    obtains the tail identity information for all test cases
    '''
    tail_d={}
    for index, ID in tqdm(enumerate(collection_keys), total=len(collection_keys)):

        tail1=tdict[(ID[0],ID[1])]
        tail2=tdict[(ID[0],ID[2])]

        tid200=score_similarities(tail1[:bp],tail2[:bp])
        tid=score_similarities(tail1[:mbp],tail2[:mbp])
        tail_over=tail_overlap(tail1, tail2)
        
        tail_d[ID]=[tid200,tid,tail_over]

    return tail_d
    
def initiation(tail_len):
    '''
    initiates overview dictionary keys
    initiates additional reference dictionaries for the analysis
    '''
    overview={
        "ID":[],
        "tail_id_200":[], "tail_id":[], "tail_overlap":[],
        
        "seq1_full":[],"seq2_full":[],

        "head_sc_200":[], "head_sc":[],

        "head1_score":[], "head2_score":[],
        "head1_score_200":[], "head2_score_200":[],
        
        "tail1_score":[], "tail2_score":[], 
        "tail1_score_200":[], "tail2_score_200":[],

        "kl-RCM-head_real":[],"kl-RCM-tail_real":[],
        "kl-RCM-head_con":[],"kl-RCM-tail_con":[],

        "D_rcm_h_real":[], "D_rcm_h_con":[],

        "D_rcm_t_real":[], "D_rcm_t_con":[] 
    }
    
    keys_ave={
    # order [which sequence, what part of it]
    "seq1_full":[(0,0),(0,1000+tail_len)],"seq2_full":[(1,0),(0,1000+tail_len)],
    "head_sc":[(2,0),(0,1000)], "head_sc_200":[(2,0),(800,1000)],
    "head1_score":[(0,1),(0,1000)],"head1_score_200":[(0,1),(800,1000)],
    "head2_score":[(1,1),(0,1000)],"head2_score_200":[(1,1),(800,1000)],
    "tail1_score":[(0,0),(1000,1000+tail_len)],"tail1_score_200":[(0,0),(1000,1200)],
    "tail2_score":[(1,0),(1000,1000+tail_len)],"tail2_score_200":[(1,0),(1000,1200)],
     }
    
    # order [seq1, gt1, seq2, gt2, seq_part, gt_part]
    keys_delta={
    "D_rcm_h":[(0,1),(2,0),(1,1),(2,0),(800,1000),(800,1000)],
    "D_rcm_t":[(0,0),(0,1),(1,0),(1,1),(1000,1200),(1000,1200)],
    }
    return overview, keys_ave, keys_delta    


def comp_init():
    '''
    initiates overview dictionary keys
    initiates additional reference dictionaries for the analysis
    '''
    
    comp_simp_keys={
        "comparison_full":["seq1_full", "seq2_full"],
        "comparison_all":["tail1_score","tail2_score"],
        "comparison_200":["tail1_score_200","tail2_score_200"],
        "comparison_head":["head1_score","head2_score"],
        "comparison_head_200":["head1_score_200","head2_score_200"]
    }
    
    comp_delta_keys={
        "RCM_head":["D_rcm_h_real","D_rcm_h_con"],
        "RCM_tail":["D_rcm_t_real", "D_rcm_t_con"]
    }
    
    comp= { "comparison_full":[], "comparison_full_n":[],
            "comparison_all":[], "comparison_all_n":[],
            "comparison_200":[], "comparison_200_n":[],
    
           "comparison_head":[], "comparison_head_n":[],
           "comparison_head_200":[], "comparison_head_200_n":[],
    
            "RCM_head":[],"RCM_tail":[], "RCM_head_n":[],"RCM_tail_n":[], 
    
            "kl-RCM-head":[],"kl-RCM-tail":[],
            "kl-RCM-head_n":[],"kl-RCM-tail_n":[]
        }
    
    return comp, comp_simp_keys, comp_delta_keys


def create_data_frame(collection, tail_det, bp=200, mbp=500):  
    '''
    creates a dataframe with the result of all methods
    '''
    def fill_ave(key, array):
        overview[key].append(math.exp(np.mean(np.log(array))))

    def fill_delta(key, array1, gt1, array2, gt2):
        overview[key+"_real"].append(np.mean(gt1-array1))
        overview[key+"_con"].append(np.mean(gt2-array2))
        

    def fill_kl(key, array1, gt1, array2, gt2):
        overview[key+"_real"].append(sum(kl_div(array1,gt1)))
        overview[key+"_con"].append(sum(kl_div(array2,gt2)))
        comparison[key].append(overview[key+"_real"][index]<overview[key+"_con"][index])
        comparison[key+"_n"].append(abs(overview[key+"_real"][index]-overview[key+"_con"][index]))
        
        
    def comp_base(key,key1,key2):
        comparison[key].append(overview[key1][index]>overview[key2][index])
        comparison[key+"_n"].append(abs(overview[key1][index]-overview[key2][index]))

    
    def comp_delta(key,key1,key2):
        comparison[key].append(overview[key1][index]<overview[key2][index])
        comparison[key+"_n"].append(abs(overview[key1][index]-overview[key2][index]))
        

    overview, keys_ave, keys_delta=initiation(mbp)
    
    comparison, keys_comp1, keys_comp2 = comp_init()

    for index, ID in tqdm(enumerate(collection.keys()), total=len(collection.keys())):
        overview["ID"].append(ID)

        overview["tail_id_200"].append(tail_det[ID][0])
        overview["tail_id"].append(tail_det[ID][1])
        overview["tail_overlap"].append(tail_det[ID][2])
        
        
        for k in keys_ave.keys():
            ind=keys_ave[k]
            fill_ave(k,collection[ID][ind[0][0]][ind[0][1]][ind[1][0]:ind[1][1]])

        for k in keys_delta.keys():
            ind=keys_delta[k]
            fill_delta(k, collection[ID][ind[0][0]][ind[0][1]][ind[4][0]:ind[4][1]],
                       collection[ID][ind[1][0]][ind[1][1]][ind[5][0]:ind[5][1]],
                       collection[ID][ind[2][0]][ind[2][1]][ind[4][0]:ind[4][1]],
                       collection[ID][ind[3][0]][ind[3][1]][ind[5][0]:ind[5][1]])
        
        fill_kl("kl-RCM-tail",
                collection[ID][0][0][1000:1200],
                collection[ID][0][1][1000:1200],
                collection[ID][1][0][1000:1200],
                collection[ID][1][1][1000:1200])
        fill_kl("kl-RCM-head",
                collection[ID][0][1][800:1000],
                collection[ID][2][0][800:1000],
                collection[ID][1][1][800:1000],
                collection[ID][2][0][800:1000])
        
        
        for c in keys_comp1.keys():
            comp_base(c,keys_comp1[c][0],keys_comp1[c][1])

        for c in keys_comp2.keys():
            comp_delta(c,keys_comp2[c][0],keys_comp2[c][1])

    output=pd.DataFrame(overview)

    for c in comparison.keys():
        output[c]=comparison[c]

    return output


def create_collection_orthologs(col_cases, col_heads):
    '''
    order: [[real_FW,real_RC], [con_FW, con_RC],
    [head_one]
    '''
    output={}
    counter=0
    for i in range(20):
        for j in range(1,20-i):
            output[(i,i+j,"first")]=[
                [np.array(col_cases[counter][0][0][0]),np.array(col_cases[counter][0][1][0])],
                [np.array(col_cases[counter][0][0][1]),np.array(col_cases[counter][0][1][1])],
                [np.array(col_heads[i])]
            ]
            output[(i,i+j,"second")]=[
                [np.array(col_cases[counter][1][0][0]),np.array(col_cases[counter][1][1][0])],
                [np.array(col_cases[counter][1][0][1]),np.array(col_cases[counter][1][1][1])],
                [np.array(col_heads[i+j])]
            ]
            counter+=1

    return output

def ortholog_tails(alignedAA,aligned):
    tail_d={}
    cuttingAA=len(alignedAA[0]["sequence"])//2
    cutting=3*(cuttingAA)
    counter=0
    for i in range(len(aligned)):
        for j in range(1,len(aligned[i:])):
            tail_1=remove_gaps(aligned[i]["sequence"][cutting:])
            tail_2=remove_gaps(aligned[i+j]["sequence"][cutting:])

            mbp=min(len(tail_1),len(tail_2))
            tid200=score_similarities(tail_1[:200],tail_2[:200])
            tid=score_similarities(tail_1[:mbp],tail_2[:mbp])
            tail_over=tail_overlap(tail_1, tail_2)
        

            tail_d[(i,i+j,"first")]=[tid200,tid,tail_over]
            tail_d[(i,i+j,"second")]=[tid200,tid,tail_over]

    return tail_d

def initiation_ortholog():
    overview={
        "ID":[],
        "tail_id_200":[], "tail_id":[], "tail_overlap":[],
        
        "seq1_full":[],"seq2_full":[],

        "head_sc_200":[], "head_sc":[],

        "head1_score":[], "head2_score":[],
        "head1_score_200":[], "head2_score_200":[],
        
        "tail1_score":[], "tail2_score":[],
        "tail1_score_200":[], "tail2_score_200":[], 
        
        "kl-RCM-head_real":[],"kl-RCM-tail_real":[],
        "kl-RCM-head_con":[],"kl-RCM-tail_con":[],

        "D_rcm_h_real":[], "D_rcm_h_con":[],

        "D_rcm_t_real":[], "D_rcm_t_con":[],
    }
    return overview

def init_each_ortho(length, cut_site):
    
    keys_ave={
    # order [which sequence, what part of it]
    "seq1_full":[(0,0),(0,length)],"seq2_full":[(1,0),(0,length)],
    "head_sc":[(2,1),(0,cut_site)], "head_sc_200":[(2,1),(cut_site-200,cut_site)],
    "head1_score":[(0,1),(0,cut_site)],"head1_score_200":[(0,1),(cut_site-200,cut_site)],
    "head2_score":[(1,1),(0,cut_site)],"head2_score_200":[(1,1),(cut_site-200,cut_site)],
    "tail1_score":[(0,0),(cut_site,length)],"tail1_score_200":[(0,0),(cut_site,cut_site+200)],
    "tail2_score":[(1,0),(cut_site,length)],"tail2_score_200":[(1,0),(cut_site,cut_site+200)],
     }
    
    # order [seq1, gt1, seq2, gt2, seq_part, gt_part]
    keys_delta={
    "D_rcm_h":[(0,1),(2,0),(1,1),(2,0),(cut_site-200,cut_site),(cut_site-200,cut_site)],
    "D_rcm_t":[(0,0),(0,1),(1,0),(1,1),(cut_site,cut_site+200),(cut_site,cut_site+200)],
    }
    return keys_ave, keys_delta

def comp_init_ortho():
    comp_simp_keys={
        "comparison_full":["seq1_full", "seq2_full"],
        "comparison_all":["tail1_score","tail2_score"],
        "comparison_200":["tail1_score_200","tail2_score_200"],
        "comparison_head":["head1_score","head2_score"],
        "comparison_head_200":["head1_score_200","head2_score_200"],
    }
    
    comp_delta_keys={
        "RCM_head":["D_rcm_h_real","D_rcm_h_con"],
        "RCM_tail":["D_rcm_t_real", "D_rcm_t_con"]
    }
    
    comp= { "comparison_full":[], "comparison_full_n":[],
            "comparison_all":[], "comparison_all_n":[], 
            "comparison_200":[], "comparison_200_n":[],

           "comparison_head":[], "comparison_head_n":[],
           "comparison_head_200":[], "comparison_head_200_n":[],
    
            "RCM_head":[],"RCM_tail":[], "RCM_head_n":[],"RCM_tail_n":[], 
    
            "kl-RCM-head":[],"kl-RCM-tail":[],
            "kl-RCM-head_n":[],"kl-RCM-tail_n":[]
        }
    
    return comp, comp_simp_keys, comp_delta_keys


def create_data_frame_ortho(collection, tail_det, bp=200):  

    def fill_ave(key, array):
        overview[key].append(math.exp(np.mean(np.log(array))))

    def fill_delta(key, array1, gt1, array2, gt2):
        overview[key+"_real"].append(np.mean(gt1-array1))
        overview[key+"_con"].append(np.mean(gt2-array2))
        

    def fill_kl(key, array1, gt1, array2, gt2):
        overview[key+"_real"].append(sum(kl_div(array1,gt1)))
        overview[key+"_con"].append(sum(kl_div(array2,gt2)))
        comparison[key].append(overview[key+"_real"][index]<overview[key+"_con"][index])
        comparison[key+"_n"].append(abs(overview[key+"_real"][index]-overview[key+"_con"][index]))
        
    def comp_base(key,key1,key2):
        comparison[key].append(overview[key1][index]>overview[key2][index])
        comparison[key+"_n"].append(abs(overview[key1][index]-overview[key2][index]))
        
    def comp_delta(key,key1,key2):
            comparison[key].append(overview[key1][index]<overview[key2][index])
            comparison[key"_n"].append(abs(overview[key1][index]-overview[key2][index]))
            
    
    overview=initiation_ortholog()
    
    comparison, keys_comp1, keys_comp2 = comp_init_ortho()

    for index, ID in tqdm(enumerate(collection.keys()), total=len(collection.keys())):

        cutsite=len(collection[ID][2][0])
        keys_ave, keys_delta=init_each_ortho(len(collection[ID][0][0]),cutsite)
        overview["ID"].append(ID)

        overview["tail_id_200"].append(tail_det[ID][0])
        overview["tail_id"].append(tail_det[ID][1])
        overview["tail_overlap"].append(tail_det[ID][2])
        
        
        for k in keys_ave.keys():
            ind=keys_ave[k]
            fill_ave(k,collection[ID][ind[0][0]][ind[0][1]][ind[1][0]:ind[1][1]])

        for k in keys_delta.keys():
            ind=keys_delta[k]
            fill_delta(k, collection[ID][ind[0][0]][ind[0][1]][ind[4][0]:ind[4][1]],
                       collection[ID][ind[1][0]][ind[1][1]][ind[5][0]:ind[5][1]],
                       collection[ID][ind[2][0]][ind[2][1]][ind[4][0]:ind[4][1]],
                       collection[ID][ind[3][0]][ind[3][1]][ind[5][0]:ind[5][1]])
        
        fill_kl("kl-RCM-tail",
                collection[ID][0][0][cutsite:cutsite+bp],
                collection[ID][0][1][cutsite:cutsite+bp],
                collection[ID][1][0][cutsite:cutsite+bp],
                collection[ID][1][1][cutsite:cutsite+bp])
        fill_kl("kl-RCM-head",
                collection[ID][0][1][cutsite-bp:cutsite],
                collection[ID][2][0][cutsite-bp:cutsite],
                collection[ID][1][1][cutsite-bp:cutsite],
                collection[ID][2][0][cutsite-bp:cutsite])
        
        
        for c in keys_comp1.keys():
            comp_base(c,keys_comp1[c][0],keys_comp1[c][1])

        for c in keys_comp2.keys():
            comp_delta(c,keys_comp2[c][0],keys_comp2[c][1])

    output=pd.DataFrame(overview)

    for c in comparison.keys():
        output[c]=comparison[c]

    return output

def get_pre_rec_data(df_data, a, k)
    pre=[]
    ret=[]
    for i in a:
        df=df_data[df_data[k+"_n"]>i]
        pre.append(df[k].to_list().count(True)/len(df))
        ret.append(len(df))
    return pre, ret
    