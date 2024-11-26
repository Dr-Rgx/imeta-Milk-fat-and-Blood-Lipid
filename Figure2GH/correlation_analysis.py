
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_corr(pd_d1, pd_d2, mtd = 'spearman'):

    if mtd not in ['pearson', 'kendall', 'spearman']:
        print ('invalid method, please denote pearson, kendall, or spearman')
        exit (-1)
  
    corr_mat = np.zeros((len(pd_d1.columns), len(pd_d2.columns)),  dtype=float)
    pd_corr_mat = pd.DataFrame(corr_mat, index=pd_d1.columns, columns=pd_d2.columns)
    for d1_i in range(len(pd_d1.columns)):
        d1_name = pd_d1.columns[d1_i]
        d1_value = pd_d1[d1_name]
        for d2_i in range(len(pd_d2.columns)):
            d2_name = pd_d2.columns[d2_i]
            d2_value = pd_d2[d2_name]
            d2_d1_corr = d1_value.corr(d2_value, method = mtd)
            corr_mat[d1_i][d2_i] = d2_d1_corr
    return corr_mat

def check_pd_data(pd_data):
    data_eng = pd_data.pow(2).sum().tolist()
    zero_loc = [i for i,v in enumerate(data_eng) if v == 0]
    if len(zero_loc) > 0:
        pd_data.drop(pd_data.columns[zero_loc], axis=1, inplace=True)
    return pd_data

if __name__ == "__main__":

    ## source data
    merged_data_path = 'data/merged.csv'  # 位置在该路径下
    X = pd.read_csv(merged_data_path)
    ## get genus and metabolism data
    genus = X.iloc[:, 252:567]
    # species = X.iloc[:, 567:618]
    metabolism = X.iloc[:, 618:-5]

    ## check data
    genus = check_pd_data(genus)
    metabolism = check_pd_data(metabolism)
    
    ## compute corr
    pd_corr_mat = compute_corr(genus, metabolism)
    
    ## save corr
    pd_corr_mat.to_csv("genus_metabolim.csv")


    


# def plot_heatmap():
#     fig, ax = plt.subplots()
#     im = ax.imshow(harvest)

#     # We want to show all ticks...
#     ax.set_xticks(np.arange(len(farmers)))
#     ax.set_yticks(np.arange(len(vegetables)))
#     # ... and label them with the respective list entries
#     ax.set_xticklabels(farmers)
#     ax.set_yticklabels(vegetables)

#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#             rotation_mode="anchor")

#     # Loop over data dimensions and create text annotations.
#     for i in range(len(vegetables)):
#         for j in range(len(farmers)):
#             text = ax.text(j, i, harvest[i, j],
#                         ha="center", va="center", color="w")

#     ax.set_title("Harvest of local farmers (in tons/year)")
#     fig.tight_layout()
#     plt.show()

