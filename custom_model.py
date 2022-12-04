import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from keras_visualizer import visualizer 
from keras.utils.vis_utils import plot_model
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import json
import os
import import_data
import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def class_proba(row):
    if row[0]>row[1]:
        return 0
    else:
        return 1


def custom_accuracy(prediction : np.array,y_test):
    y_hat_values = np.apply_along_axis(class_proba, 1, prediction)
    y_test_values = np.apply_along_axis(class_proba,1,y_test.values)
    res = np.equal(y_hat_values,y_test_values)
    count = np.bincount(res)
    return count,len(res),count/len(res)

def train_model_from_pipeline(model = None,epoch = 600,batch_size = 1000,plot_accuracy = False,test_ratio = 0.1,prt=True):
    
    frames,_ = import_data.main_pipeline(feat_function=6,test_ratio=test_ratio)
    x_train,y_train,x_test,y_test = frames
    model = model if model else create_model(input_shape=x_train.shape[1])

    history = model.fit(x_train.to_numpy(), y_train.to_numpy(),epochs= epoch ,batch_size =batch_size,verbose=0)

    if plot_accuracy:
        h_values = history.history['accuracy']
        h_delta = [np.log(h_values[i+1]/h_values[i]) for i in range(len(h_values)-1)]
        fig,axs = plt.subplots(2,1,figsize=(15,8),sharex = True)
        ax1 = axs[0]
        ax1.plot(h_values)
        ax1.set_title('Model accuracy')
        ax1.set_ylabel('accuracy')
        ax1.legend(['train accuracy'], loc='upper left')
        ax2 = axs[1]
        ax2.plot(h_delta,color="orange")
        ax2.plot([0 for _ in range(len(h_delta))], color = "b",linestyle="dashed")
        ax2.set_title('Accuracy Delta')
        ax2.set_ylabel('$\Delta_{Accuracy}$')
        ax2.legend(['train delta'], loc='upper left')
        ax2.set_xlabel('epoch')
        fig.savefig(f"output/model_train_b{batch_size}_e{epoch}.png")
    
    _, accuracy = model.evaluate(x_test.to_numpy(), y_test.to_numpy())
    prediction = model.predict(x_test.to_numpy())
    _,_,(_,a) = custom_accuracy(prediction,y_test)

    if prt:
        print(f"Accuracy : {round(accuracy,4)*100}% || {a}")

    return accuracy    
    
def create_model(plot=False,input_shape=86):

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_shape,)),
        keras.layers.Dense(100, activation="tanh",name="Hidden_layer_1"),
        keras.layers.Dense(100, activation="tanh",name="Hidden_layer_2"),
        keras.layers.Dense(100, activation="tanh",name="Hidden_layer_3"),
        keras.layers.Dense(2, activation="softmax",name="Output")
    ])
    model.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    if plot:
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model


def full_train(model=None,epoch = 600,batch_size = 1000,plot_accuracy = False):
    frames,_ = import_data.main_pipeline(feat_function=6,test_ratio=0,split=False)
    X = frames.drop(["LABEL_DOWN","LABEL_UP"],axis=1)
    Y = frames[["LABEL_DOWN","LABEL_UP"]]
    model = model if model else create_model(input_shape=X.shape[1])
    history = model.fit(X.to_numpy(), Y.to_numpy(),epochs= epoch ,batch_size =batch_size,verbose=0)
    if plot_accuracy:
        h_values = history.history['accuracy']
        h_delta = [np.log(h_values[i+1]/h_values[i]) for i in range(len(h_values)-1)]
        fig,axs = plt.subplots(2,1,figsize=(15,8),sharex = True)
        ax1 = axs[0]
        ax1.plot(h_values)
        ax1.set_title('Model accuracy')
        ax1.set_ylabel('accuracy')
        ax1.legend(['train accuracy'], loc='upper left')
        ax2 = axs[1]
        ax2.plot(h_delta,color="orange")
        ax2.plot([0 for _ in range(len(h_delta))], color = "b",linestyle="dashed")
        ax2.set_title('Accuracy Delta')
        ax2.set_ylabel('$\Delta_{Accuracy}$')
        ax2.legend(['train delta'], loc='upper left')
        ax2.set_xlabel('epoch')
        fig.savefig(f"output/full_model_train_b{batch_size}_e{epoch}.png")
    print(f"Accuracy : {round(history.history['accuracy'][-1],4)*100}%")
    return model

def full_predict(model,X_test=None,output_path= "data/Data_B_pred.csv"):

    if X_test is None:

        X_test = import_data.import_a_set(path="data/Data_B_nolabels.csv",is_test=True)
        X_test = import_data.rescale_features_6(frame=X_test,no_label=True)
    
    Y_test = model.predict(X_test)
    Y_test_normal = np.apply_along_axis(class_proba,1,Y_test)

    df_out = pd.DataFrame(Y_test_normal,columns=["LABEL"])
    df_out.to_csv(output_path,header=False,index=False)

def grid_search_b_e():
    frames,_ = import_data.main_pipeline(feat_function=6,test_ratio=0,split=False)
    X = frames.drop(["LABEL_DOWN","LABEL_UP"],axis=1)
    Y = frames[["LABEL_DOWN","LABEL_UP"]]

    model = KerasClassifier(model=create_model, verbose=0)
    # define the grid search parameters
    #batch_size = [16,64,512,1024,2048]
    #epochs = [50,250,600,1200]
    batch_size = [1024,2048]
    epochs = (250,600)
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=2)
    grid_result = grid.fit(X, Y)
    return grid_result

def custom_grid_search(batch_epoch_dict,test_ratio=0.1,number_test = 1):

    cv_results_df = pd.DataFrame(columns = ["mean_acc","std_dev_acc","batch_size","epochs"])

    for b,epochs_list in batch_epoch_dict.items():
        for epochs in epochs_list:
            res = []
            for _ in range(number_test):
                m = create_model()
                acc = train_model_from_pipeline(m,epoch= epochs,batch_size=b,test_ratio=test_ratio,prt=False)
                res.append(acc)
            cv_results_df = cv_results_df.append({"mean_acc":np.array(res).mean(),"std_dev_acc":np.array(res).std(),"batch_size" :b,"epochs":epochs},ignore_index=True)
    
    with open("cv_result_x.json","w") as f:
        f.write(cv_results_df.to_json())

if __name__=="__main__":
    full_predict(full_train(batch_size=1000,epoch=650,plot_accuracy=True))
