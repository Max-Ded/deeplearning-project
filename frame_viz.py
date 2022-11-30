import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import import_data


def level_fig(frame,col,col_label,suffix="",bins=50):
    side = ["A","B"]
    color=["blue","red"]
    side_label=["ASK","BID"]
    fig1, axs = plt.subplots(2,5,figsize=(40,10))
    fig2,ax2 = plt.subplots(1,1,figsize=(40,40))
    for i in range(2):
        for j in range(4):
            ax = axs[i,j]
            ax.set_title(f"{side_label[i]} {col_label} - Level : {j+1}")
            ax.set_xlabel(col_label)
            plt.xticks(rotation = 45)
            frame[f"{side[i]}_{col}_L{j+1}{suffix}"].plot(kind="hist",bins=bins,color=color[i],ax=ax)
            frame[f"{side[i]}_{col}_L{j+1}{suffix}"].plot(kind="hist",bins=bins,color=color[i],alpha=0.2,ax=axs[i,4])
            frame[f"{side[i]}_{col}_L{j+1}{suffix}"].plot(kind="hist",bins=bins,color=color[i],alpha=0.2,ax=ax2)
    axs[0,4].set_title("ASK {col_label} - all levels")
    axs[1,4].set_title("BID {col_label} - all levels")
    fig1.savefig(f"image/AB_{col_label}{suffix}_plots.png")
    fig2.savefig(f"image/AB_{col_label}{suffix}_all_in_one.png")
    

if __name__=="__main__":

    #frame = pd.read_excel("output/processed_data.xlsx")
    #frame = import_data.import_a_set(path="data/Data_B_nolabels.csv",is_test=True)
    frame,_ = import_data.main_pipeline(feat_function=6,split=False)
    level_fig(frame,"P","Prices",suffix="_UM")
    #level_fig(frame,"V","Volumes",bins=500)
