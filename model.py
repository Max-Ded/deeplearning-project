import pandas as pd
import tensorflow.keras as keras
import numpy as np
import import_data
import matplotlib.pyplot as plt

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
    frames,params = import_data.main_pipeline(feat_function=4,test_ratio=0.1)
    x_train,y_train,x_test,y_test = frames
    training_point,input_shape = params

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_shape,)),
        keras.layers.Dense(input_shape+15, activation="relu"),
        keras.layers.Dense(input_shape+15, activation="relu"),
        keras.layers.Dense(input_shape+15, activation="relu"),
        keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train.to_numpy(), y_train.to_numpy(),epochs=500,batch_size = training_point//100,verbose=0)
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    _, accuracy = model.evaluate(x_test.to_numpy(), y_test.to_numpy())
    print(f"Accuracy : {round(accuracy,4)*100}%")

    prediction = model.predict(x_test.to_numpy())
    y_hat = pd.DataFrame(prediction)

    _,_,(_,a) = custom_accuracy(prediction,y_test)
    print(a)
    plt.show()
