from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from imblearn.under_sampling import ClusterCentroids

class Preprocessor:
    def __init__(self, fraud_df):
        self.fraud_df = fraud_df

    def normalize(self):
        ## Normalizing the data to avoid misleading the model
        std_scaler = StandardScaler()
        self.fraud_df['normalizedAmount'] = std_scaler.fit_transform(self.fraud_df['Amount'].values.reshape(-1,1))
        self.fraud_df = self.fraud_df.drop(['Amount'],axis=1)
        ## Time is irrelevant, as it seemed from the descriptive analysis, so dropping it
        self.fraud_df['normalizedTime'] = std_scaler.fit_transform(self.fraud_df['Time'].values.reshape(-1,1))
        self.fraud_df = self.fraud_df.drop(['Time'],axis=1)

    def split_X_Y(self):
        fraud_df_X = self.fraud_df.drop(['Class'], axis = 1)
        fraud_df_Y = self.fraud_df['Class']
        return fraud_df_X, fraud_df_Y

    def ApplySMOTE(self, random_state, fraud_df_X, fraud_df_Y):
        sm = SMOTE(random_state = random_state)
        X_res, y_res = sm.fit_resample(fraud_df_X, fraud_df_Y)
        return X_res, y_res

    def ApplyClusterCentroids(self, random_state, fraud_df_X, fraud_df_Y):
        cluster_centroids = ClusterCentroids(random_state = 42)
        X_underSam, y_underSam = cluster_centroids.fit_resample(fraud_df_X, fraud_df_Y)
        return X_underSam, y_underSam

    def Shuffle_data(self, X, y):
        X, y = shuffle(X, y)
        return X, y

