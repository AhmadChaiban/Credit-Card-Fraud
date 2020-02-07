import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

class NNClassifier:
    def __init__(self, input, dense1, dense2, dense3, dense4, shape_X_train):

        self.model = keras.Sequential([
            keras.layers.Dense(input_shape = shape_X_train),
            keras.layers.Dense(units=256, activation='relu'),
            keras.layers.Dense(units=192, activation='relu'),
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dense(units=2, activation='softmax')
        ])

    def data_set_creator(self, X_train, y_train, number_of_classes):
        y_train = tf.one_hot(y_train, depth=number_of_classes)
        return tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(128)

    def compile(self, model, optimizer, loss, accuracy_metric):
        model.compile(optimizer='adam',
                      loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    def train(self, X_train, y_train):
        data = self.data_set_creator(X_train, y_train, 2)
        history = self.model.fit(X_train)


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




