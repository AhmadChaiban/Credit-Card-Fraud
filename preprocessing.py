from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, fraud_df):
        self.fraud_df = fraud_df

    def normalize(self):
        ## Normalizing the data to avoid misleading the model
        self.fraud_df['normalizedAmount'] = StandardScaler().fit_transform(self.fraud_df['Amount'].values.reshape(-1,1))
        self.fraud_df = self.fraud_df.drop(['Amount'],axis=1)
        ## Time is irrelevant, as it seemed from the descriptive analysis, so dropping it
        self.fraud_df = self.fraud_df.drop(['Time'],axis=1)

    def split_X_Y(self):
        fraud_df_X = self.fraud_df.drop(['Class'], axis = 1)
        fraud_df_Y = self.fraud_df['Class']
        return fraud_df_X, fraud_df_Y

    def ApplySMOTE(self, random_state, fraud_df_X, fraud_df_Y):
        sm = SMOTE(random_state = random_state)
        X_res, y_res = sm.fit_resample(fraud_df_X, fraud_df_Y)
        return X_res, y_res