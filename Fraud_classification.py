import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from preprocessing import Preprocessor
from tensorflow.keras import optimizers
from sklearn.manifold import TSNE

class NNClassifier:
    def __init__(self, input, dense1, dense2, dense4):
        self.model = keras.Sequential([
            keras.layers.Dense(units = dense1, input_dim = input, activation = 'relu'),
            keras.layers.Dense(units=dense2, activation='relu'),
            keras.layers.Dense(units=dense4, activation='sigmoid')
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

    def train(self, X_train, y_train, epochs, batch_size, validation_split):
        X_train, y_train = self.convert_to_tensor(X_train, y_train)
        # data = self.data_set_creator(X_train, y_train, 1)
        # print(data.shape)
        history = self.model.fit(x = X_train,
                                 y = y_train,
                                 epochs = epochs,
                                 batch_size = batch_size,
                                 validation_split=validation_split)
        return history

    def prediction_adjustor(self, y_pred):
        y_pred_new = []
        for i in range(len(y_pred)):
            if y_pred[i][0] >0.5:
                y_pred_new.append(1)
            else:
                y_pred_new.append(0)
        print(y_pred_new)
        return y_pred_new

    def predict(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print(y_pred)
        y_pred_adjusted = self.prediction_adjustor(y_pred)
        print(y_pred_adjusted)
        print(y_test)
        print(y_test.shape, np.array(y_pred_adjusted).shape)
        return np.array(y_pred_adjusted).reshape([len(y_pred_adjusted),1]).T, \
               accuracy_score(y_test, y_pred_adjusted), y_pred

    def plot_roc_curve(self, y_test, y_pred):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr)
        plt.title('ROC Curve')
        plt.show()

    def plot_loss(self, history):
        plt.title('Loss')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.show()

    def plot_accuracy(self, history):
        plt.title('Accuracy')
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.show()

if __name__ == '__main__':
    ## Importing the data
    print("Reading from Database...")
    fraud_df_X = pd.read_csv('under-sampled-data.csv').drop(['Unnamed: 0'], axis = 1)
    print(fraud_df_X.head())
    fraud_df_Y = pd.read_csv('under-sampled-data-class.csv').drop(['Unnamed: 0'], axis = 1)
    print(fraud_df_Y.head())
    print('Applying t-SNE...')
    ## Applying t-SNE to the data
    X_res_embedded = TSNE(n_components = 2, random_state = 0).fit_transform(fraud_df_X)
    print('done!')
    ## Creating the preprocessor
    preprocessor = Preprocessor(X_res_embedded)
    ## Reshuffling the data
    X_res_shuffled, y_res_shuffled = preprocessor.Shuffle_data(fraud_df_X, fraud_df_Y)
    ## Train test split
    print('Final features for training:')
    X_train, X_test, y_train, y_test = train_test_split(X_res_shuffled, y_res_shuffled, test_size=0.20, random_state=42)
    print(pd.DataFrame(X_train).head())
    ## Defining the classifier
    model = NNClassifier(30, 15, 15, 1)
    ## Compoling the model
    model.compile(optimizer= optimizers.SGD(learning_rate = 0.001),
                  loss = tf.losses.MeanSquaredError(),
                  accuracy_metric = ['accuracy'] )
    ## Training and recording history
    history = model.train(X_train, y_train, epochs = 500,  batch_size = 32, validation_split = 0.2)
    ## Predicting on the test set
    y_pred, accuracy, y_pred_proba = model.predict(X_test, y_test)
    ## Showing the accuracy of the model
    print(f"Accuracy: {accuracy}")
    ## confusion matrix
    print(confusion_matrix(np.array(y_test).T.reshape([len(y_test),]), y_pred.reshape([len(y_pred.T),])))
    ## Getting the true negatives, false positives, false negatives and false positives count
    tn, fp, fn, tp = confusion_matrix(np.array(y_test).T.reshape([len(y_test),]), y_pred.reshape([len(y_pred.T),])).ravel()
    print(f'True Negatives {tn}, False Positives {fp}, False Negatives {fn}, True Positives {tp}')
    ## Plotting the recorded history of training and validation loss
    model.plot_loss(history)
    model.plot_accuracy(history)
    model.plot_roc_curve(y_test, y_pred_proba)





