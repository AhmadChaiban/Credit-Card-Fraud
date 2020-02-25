import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing import Preprocessor
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix

class NNClassifier:
    def __init__(self, input, dense1, dense2, dense3, dense4):
        self.model = keras.Sequential([
            keras.layers.Dense(units = input, input_dim = input, activation = 'relu'),
            keras.layers.Dense(units=dense1, activation='relu'),
            keras.layers.Dense(units=dense2, activation='relu'),
            keras.layers.Dense(units=dense3, activation='relu'),
            keras.layers.Dense(units=dense4, activation='softmax')
        ])
        
    # def data_set_creator(self, X_train, y_train, number_of_classes):
    #     y_train = tf.one_hot(y_train, depth=number_of_classes)
    #     return tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(128)

    def convert_to_tensor(self, X_train, y_train):
        x = tf.cast(np.array(X_train), tf.float32)
        y = tf.cast(np.array(y_train), tf.int64)
        return x,y

    def compile(self, optimizer, loss, accuracy_metric):
        self.model.compile(optimizer= optimizer,
                           loss= loss,
                           metrics= accuracy_metric)

    def train(self, X_train, y_train, epochs):
        X_train, y_train = self.convert_to_tensor(X_train, y_train)
        print(X_train.shape)
        # data = self.data_set_creator(X_train, y_train, 1)
        # print(data.shape)
        history = self.model.fit(x = X_train,
                                 y = y_train,
                                 epochs = epochs,
                                 batch_size = 128,
                                 validation_split=0.2)
        return history

    def prediction_adjustor(self, y_pred):
        y_pred_new = []
        for i in range(len(y_pred)):
            if y_pred[i][0] == 1:
                y_pred_new.append(1)
            else:
                y_pred_new.append(0)
        return y_pred_new

    def predict(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_adjusted = self.prediction_adjustor(y_pred)
        return y_pred_adjusted, accuracy_score(np.array(y_test).T, y_pred_adjusted)

    def plot_loss(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.show()

    def plot_accuracy(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.show()

if __name__ == '__main__':
    ## Importing the data
    print("Reading from Database...")
    fraud_df = pd.read_csv('creditcard.csv')
    print(fraud_df.head())
    preprocessor = Preprocessor(fraud_df)
    ## Normalizing the data and deleting some irrelevant data like time
    preprocessor.normalize()
    ## Separating the features from the labels
    fraud_df_X, fraud_df_Y = preprocessor.split_X_Y()
    ## Oversampling the data
    X_res, y_res = preprocessor.ApplySMOTE(42, fraud_df_X, fraud_df_Y)
    ## Train test split
    print('Final features for training:')
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=42)
    print(X_train.head())
    ## Defining the classifier
    model = NNClassifier(29, 32, 20, 10, 2)
    ## Compoling the model
    model.compile(optimizer= optimizers.SGD(learning_rate = 0.01),
                  loss = tf.losses.CategoricalCrossentropy(from_logits=True),
                  accuracy_metric = ['accuracy'] )
    ## Training and recording history
    history = model.train(X_train, y_train, 5)
    ## Predicting on the test set
    y_pred, accuracy = model.predict(X_test, y_test)
    ## Showing the accuracy of the model
    print(accuracy)
    ## confusion matrix
    print(confusion_matrix(np.array(y_test).T, y_pred))
    ## Plotting the recorded history of training and validation loss
    model.plot_loss(history)
    model.plot_accuracy(history)





