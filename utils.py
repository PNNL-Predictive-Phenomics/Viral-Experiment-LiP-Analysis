import sys, os, re, warnings, copy, math
warnings.filterwarnings("ignore")

from IPython.display import display
from pathlib import Path
home = str(Path.home())

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [7, 5]
# plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
import seaborn as sns
import plotnine

from Bio import SeqIO
from Bio.PDB import *
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.MMCIFParser import MMCIFParser
import networkx as nx
# import nglview as nv

def nip_off_pept(peptide):
    pept_pattern = "\.(.+)\."
    subpept = re.search(pept_pattern, peptide).group(1)
    return(subpept)
def strip_peptide(peptide, nip_off=True):
    if nip_off:
        return(re.sub(r"[^A-Za-z]+", '', nip_off_pept(peptide)))
    else:
        return(re.sub(r"[^A-Za-z]+", '', peptide))
def get_ptm_pos_in_pept(peptide, ptm_label = '*', special_chars = r'.]+-=@_!#$%^&*()<>?/\|}{~:['):
    peptide = nip_off_pept(peptide)
    if ptm_label in special_chars:
        ptm_label = '\\' + ptm_label
    ptm_pos = [m.start() for m in re.finditer(ptm_label, peptide)]
    pos = sorted([val - i - 1 for i, val in enumerate(ptm_pos)])
    return(pos)
def get_yst(strip_pept, ptm_aa = "YSTyst"):
    return([[i, letter.upper()] for i, letter in enumerate(strip_pept) if letter in ptm_aa])
def get_ptm_info(peptide, residue = None, prot_seq = None, ptm_label = '*'):
    if prot_seq != None:
        clean_pept = strip_peptide(peptide)
        pept_pos = prot_seq.find(clean_pept)
        all_yst = get_yst(clean_pept)
        all_ptm = [[pept_pos + yst[0] + 1, yst[1], yst[0]] for yst in all_yst]
        return(all_ptm)
    if residue != None:
        subpept = nip_off_pept(peptide)
        split_substr = subpept.split(ptm_label)
        res_pos = sorted([int(res) for res in re.findall(r'\d+', residue)])
        first_pos = res_pos[0]
        res_pos.insert(0, first_pos - len(split_substr[0]))
        pept_pos = 0
        all_ptm = []
        for i, res in enumerate(res_pos):
            # print(i)
            if i > 0:
                pept_pos += len(split_substr[i-1])
            yst_pos = get_yst(split_substr[i])
            if len(yst_pos) > 0:
                for j in yst_pos:
                    ptm = [j[0] + res_pos[i] + 1, j[1], pept_pos + j[0]]
                    all_ptm.append(ptm)
        return(all_ptm)
def relable_pept(peptide, label_pos, ptm_label = '*'):
    strip_pept = strip_peptide(peptide)
    for i, pos in enumerate(label_pos):
        strip_pept = strip_pept[:(pos + i + 1)] + ptm_label + strip_pept[(pos + i + 1):]
    return(peptide[:2] + strip_pept + peptide[-2:])
def get_phosphositeplus_pos(mod_rsd):
    return([int(re.sub(r"[^0-9]+", '', mod)) for mod in mod_rsd])
def get_res_names(residues):
    res_names = [[res for res in re.findall(r'[A-Z]\d+[a-z\-]+', residue)] if residue[0] != 'P' else [residue] for residue in residues]
    return(res_names)
def get_res_pos(residues):
    res_pos = [[int(res) for res in re.findall(r'\d+', residue)] if residue[0] != 'P' else [0] for residue in residues]
    return(res_pos)


def plot_barcode(pal, ticklabel = None, barcode_name = None, ax=None, size = (10,2)):
    """Plot the values in a color palette as a horizontal array.
    Parameters
    ----------
    pal : sequence of matplotlib colors
        colors, i.e. as returned by seaborn.color_palette()
    size :
        figure size of plot
    ax :
        an existing axes to use
    """

    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    n = len(pal)
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=size)
    ax.imshow(np.arange(n).reshape(1, n),
              cmap=mpl.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.set_yticks([0])
    ax.set_yticklabels([barcode_name])
    # The proper way to set no ticks
    # ax.yaxis.set_major_locator(ticker.NullLocator())
    # ax.set_xticks(np.arange(n) - .5)
    # ax.set_xticks(np.arange(n))
    ax.set_xticks(np.arange(0,n,np.ceil(n/len(ticklabel)).astype("int")))
    # Ensure nice border between colors
    # ax.set_xticklabels(["" for _ in range(n)])
    ax.set_xticklabels(ticklabel)
    # return(ax)


def get_barcode(fc_bar, color_levels = 20, fc_bar_max = None):
    # fc_bar = copy.deepcopy(res_fc_diff[["FC_DIFF", "FC_TYPE", "Res"]])
    both_pal_vals = sns.color_palette("Greens", color_levels)
    up_pal_vals = sns.color_palette("Reds", color_levels)
    down_pal_vals = sns.color_palette("Blues", color_levels)
    insig_pal_vals = sns.color_palette("Greys", color_levels)
    if fc_bar_max == None:
        fc_bar_max = fc_bar["FC_DIFF"].abs().max()
    bar_code = []
    for i in range(fc_bar.shape[0]):
        if fc_bar.iloc[i, 1] == "both":
            bar_code.append(both_pal_vals[np.ceil(abs(fc_bar.iloc[i, 0])/fc_bar_max * color_levels).astype("int") - 1])
        elif fc_bar.iloc[i, 1] == "up":
            bar_code.append(up_pal_vals[np.ceil(abs(fc_bar.iloc[i, 0])/fc_bar_max * color_levels).astype("int") - 1])
        elif fc_bar.iloc[i, 1] == "down":
            bar_code.append(down_pal_vals[np.ceil(abs(fc_bar.iloc[i, 0])/fc_bar_max * color_levels).astype("int") - 1])
        elif fc_bar.iloc[i, 1] == "insig":
            bar_code.append(insig_pal_vals[np.ceil(abs(fc_bar.iloc[i, 0])/fc_bar_max * color_levels).astype("int") - 1])
        else:
            bar_code.append((0,0,0))
    return(bar_code)

    
def get_protein_res(proteome, uniprot_id, prot_seqs):
    protein = proteome[proteome["uniprot_id"] == uniprot_id]
    protein.reset_index(drop=True, inplace=True)
    prot_seq_search = [seq for seq in prot_seqs if seq.id == uniprot_id]
    prot_seq = prot_seq_search[0]
    sequence = str(prot_seq.seq)
    seq_len = len(sequence)
    # print(seq_len)
    clean_pepts = [strip_peptide(pept) for pept in protein["peptide"].to_list()]
    protein["clean_pept"] = clean_pepts 
    pept_start = [sequence.find(clean_pept) for clean_pept in clean_pepts]
    pept_end = [sequence.find(clean_pept) + len(clean_pept) for clean_pept in clean_pepts]
    protein["pept_start"] = pept_start
    protein["pept_end"] = pept_end
    protein["residue"] = [[res + str(sequence.find(clean_pept)+i) for i, res in enumerate(clean_pept)] for clean_pept in clean_pepts]
    protein_res = protein.explode("residue")
    protein_res.reset_index(drop=True, inplace=True)
    return(protein_res)
    
def adjusted_p_value(pd_series, ignore_na = True, filling_val = 1):
    output = pd_series.copy()
    if pd_series.isna().sum() > 0:
        # print("NAs present in pd_series.")
        if ignore_na:
            print("Ignoring NAs.")
            # pd_series = 
        else:
            # print("Filling NAs with " + str(filling_val))
            output = sp.stats.false_discovery_control(pd_series.fillna(filling_val))
    else:
        # print("No NAs present in pd_series.")
        output = sp.stats.false_discovery_control(pd_series)
    return(output)
            
