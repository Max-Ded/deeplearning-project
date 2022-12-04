import pandas as pd
from sklearn.decomposition import PCA
import import_data
import matplotlib.pyplot as plt
import numpy as np 
from collections import Counter

def PCA_dataframe():
    df,_ = import_data.main_pipeline(feat_function=0,split=False)
    df = df.drop("LABEL",axis=1)
    # normalize data
    from sklearn import preprocessing
    data_scaled = pd.DataFrame(preprocessing.scale(df),columns = df.columns) 
    # PCA
    n_components = 10
    pca = PCA(n_components=n_components)
    pca.fit_transform(data_scaled)

    # Dump components relations with features:
    print(pd.DataFrame(pca.components_,columns=data_scaled.columns,index = [f"PC-{i}" for i in range(1,n_components+1)]))

def measure_random(sample):
    score =0
    for index in range(len(sample)-1):
        if sample[index]!=sample[index+1]:
            score+=1
    return score

def measure_entropy(sample):
    c = Counter(sample)
    e = 0
    for v in c.values():
        e-= v/len(sample) * np.log(v/len(sample))
    return e 

def most_recent_tick():
    df,_ = import_data.main_pipeline(feat_function=0,split=False)

    df = df.drop([c for c in df.columns if ("LC" not in c and c!="LABEL")],axis=1)
    df["CONCAT"] = df["LC_1"].apply(str) + df["LC_2"].apply(str) +  df["LC_3"].apply(str)  + df["LC_4"].apply(str) + df["LC_5"].apply(str)
    df = df.drop([c for c in df.columns if "LC" in c],axis = 1)
    bins = df["LABEL"].unique()
    df_0 = df[df["LABEL"]==0].drop("LABEL",axis=1).value_counts(sort=True)
    df_1 = df[df["LABEL"]==1].drop("LABEL",axis=1).value_counts(sort=True)

    df_count = pd.DataFrame(columns=["0","1"])
    df_count["0"] = df_0
    df_count["1"] = df_1
    df_count['ratio'] = (df_count["0"]/df_count["1"]).apply(lambda x : max(x,1/x))
    df_count = df_count.sort_values("ratio",ascending=False).drop("ratio",axis=1)

    df_count.index = [f"{k[0]}" for k in list(df_count.index)]

    fig = plt.figure(figsize=(25, 8))
    ax = fig.add_subplot(111)
    #ax.plot([k for k in df_count.index],[round(measure_entropy(k),2) for k in df_count.index],color="red",linestyle="dashed",label="Entropy")
    #ax.plot([k for k in df_count.index],[round(measure_random(k),2) for k in df_count.index],color="blue",linestyle="dashed",label="Random")

    bar_plot =df_count.plot(ax=ax,kind='bar', rot=0, xlabel='Serie', ylabel='Value', title='Distribution of Labels',secondary_y=True)
    # add some labels
    for c in ax.containers:
        # set the bar label
        ax.bar_label(c, fmt='%.0f', label_type='edge')
    ax.tick_params(axis='x', labelrotation=45)

    # move the legend out of the plot
    #ax.legend(handles= [bar_plot],title='Columns', bbox_to_anchor=(1, 1.02), loc='upper left')
    ax.legend()
    plt.show()

    fig.savefig(fname="image/count_tick_serie_simple.png")


if __name__=="__main__":
    most_recent_tick()