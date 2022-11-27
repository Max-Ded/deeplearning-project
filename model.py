import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras as keras

import import_data


def class_proba(row):
    if row[0]>row[1]:
        return -1
    else:
        return 1


def custom_accuracy(prediction : np.array,y_test):
    y_hat_values = np.apply_along_axis(class_proba, 1, prediction)
    y_test_values = np.apply_along_axis(class_proba,1,y_test.values)
    res = np.equal(y_hat_values,y_test_values)
    count = np.bincount(res)
    return count,len(res),count/len(res)

if __name__=="__main__":
    frames,params = import_data.main_pipeline(feat_function=5,test_ratio=0.1)
    x_train,y_train,x_test,y_test = frames
    training_point,input_shape = params

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_shape,)),
        keras.layers.Dense(int(input_shape+15), activation="tanh"),
        keras.layers.Dense(int(input_shape+15), activation="tanh"),
        keras.layers.Dense(int(input_shape+15), activation="tanh"),
        keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train.to_numpy(), y_train.to_numpy(),epochs=1500,batch_size = training_point//100,verbose=0)
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
    
    _, accuracy = model.evaluate(x_test.to_numpy(), y_test.to_numpy())
    print(f"Accuracy : {round(accuracy,4)*100}%")

    prediction = model.predict(x_test.to_numpy())
    y_hat = pd.DataFrame(prediction)

    _,_,(_,a) = custom_accuracy(prediction,y_test)
    print(a)
    plt.show()
