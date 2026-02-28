import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):

    # Drop unnecessary columns
    df = df.drop(['CustomerId', 'Surname'], axis=1)

    # Encode Gender
    df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})

    # One-hot encode Geography
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

    return df