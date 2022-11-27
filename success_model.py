from numpy import loadtxt
import tensorflow.keras as keras
import import_data


def model_1():
    """
    78.12% Accuracy
    Doesn't seem to influenced by shape of network
    """
    dataset = import_data.import_a_set()
    dataset = import_data.rescale_features_1(dataset)
    x_train,y_train,x_test,y_test = import_data.split_training_data(dataset,test_ratio=0.1)

    training_point,input_shape = x_train.shape

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_shape,)),
        keras.layers.Dense(input_shape+15, activation="relu"),
        keras.layers.Dense(input_shape+15, activation="relu"),
        keras.layers.Dense(input_shape+15, activation="relu"),
        keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train.to_numpy(), y_train.to_numpy(),epochs=1500,batch_size = training_point//100)
    _, accuracy = model.evaluate(x_test.to_numpy(), y_test.to_numpy())
    print(f"Accuracy : {round(accuracy,4)*100}%")

def model_2():

    frames,params = import_data.main_pipeline(feat_function=3)
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
    model.fit(x_train.to_numpy(), y_train.to_numpy(),epochs=150,batch_size = training_point//100)
    _, accuracy = model.evaluate(x_test.to_numpy(), y_test.to_numpy())
    print(f"Accuracy : {round(accuracy,4)*100}%")
