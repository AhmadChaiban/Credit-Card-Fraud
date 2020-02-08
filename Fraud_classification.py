import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NNClassifier:
    def __init__(self, input, dense1, dense2, dense3, dense4):

        self.model = keras.Sequential([
            keras.layers.Dense(units = input),
            keras.layers.Dense(units=dense1, activation='relu'),
            keras.layers.Dense(units=dense2, activation='relu'),
            keras.layers.Dense(units=dense3, activation='relu'),
            keras.layers.Dense(units=dense4, activation='softmax')
        ])

    def data_set_creator(self, X_train, y_train, number_of_classes):
        y_train = tf.one_hot(y_train, depth=number_of_classes)
        return tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(128)

    def convert_to_tensor(self, X_train, y_train):
        x = tf.cast(np.array(X_train), tf.float32)
        y = tf.cast(np.array(y_train), tf.int64)
        return x,y

    def compile(self, model, optimizer, loss, accuracy_metric):
        self.model.compile(optimizer= optimizer,
                           loss= loss,
                           metrics= accuracy_metric)

    def train(self, X_train, y_train, epochs):
        X_train, y_train = self.convert_to_tensor(X_train, y_train)
        data = self.data_set_creator(X_train, y_train, 1)
        history = self.model.fit(data,
                                 epochs = epochs)
                                 # validation_split = validation_split)
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
        return accuracy_score(np.array(y_test).T, y_pred_adjusted)

    def plot_history(self, history):
        plt.plot(history)
        plt.show()

if __name__ == '__main__':
    ## Importing the data
    fraud_df = pd.read_csv('creditcard.csv')
    ## Splitting into X and Y
    fraud_df_X = fraud_df.drop(['Class'], axis = 1)
    fraud_df_Y = fraud_df['Class']
    ## SMOTE Oversampling
    sm = SMOTE(random_state = 42)
    X_res, y_res = sm.fit_resample(fraud_df_X, fraud_df_Y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=42)

    model = NNClassifier(3, 300, 200, 50, 2)

    model.compile(model, optimizer='adam',
                  loss = tf.losses.CategoricalCrossentropy(from_logits=True),
                  accuracy_metric = ['accuracy'] )

    history = model.train(X_train, y_train, 20)
    accuracy = model.predict(X_test, y_test)
    print(accuracy)




