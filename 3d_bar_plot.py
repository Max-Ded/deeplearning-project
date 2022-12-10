from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_json("cv_result_x_cloud.json")
df_ = pd.read_json("cv_result_x.json")

colors = plt.cm.Spectral(df["mean_acc"].values/df["mean_acc"].values.max())


batch_size = sorted(list(df["batch_size"].unique()))
batch_dict = {b:2*i for i,b in enumerate(batch_size)}

epochs = sorted(list(df["epochs"].unique()))
epoch_dict = {e:2*i for i,e in enumerate(epochs)}

x = []
y= []
z = []
dx = []
dy = []
dz = []
for _,row in df.iterrows():
    x.append(batch_dict.get(row["batch_size"]))
    y.append(epoch_dict.get(row["epochs"]))
    z.append(0)
    dx.append(1)
    dy.append(1)
    dz.append(row["mean_acc"])

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(projection='3d')
bb = ax.bar3d(x, y, z, dx, dy, dz, color=colors,cmap = plt.cm.Spectral)

ax.set_xticks(list(batch_dict.values()))
ax.set_yticks(list(epoch_dict.values()))
ax.set_xticklabels(batch_size)
ax.set_yticklabels(epochs)


ax.set_title("Grid search of the optimal batch-size/epochs combination")
ax.set_xlabel('Batch size')
ax.set_ylabel('Epochs')
ax.set_zlabel('Mean accuracy')

#plt.colorbar(bb,label="Accuracy", orientation="vertical",ax=ax)

plt.show()
