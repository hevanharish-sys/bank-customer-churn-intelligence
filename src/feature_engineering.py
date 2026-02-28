def add_features(df):

    df['BalanceToSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['ProductDensity'] = df['NumOfProducts'] / (df['Tenure'] + 1)
    df['AgeTenureInteraction'] = df['Age'] * df['Tenure']
    df['EngagementScore'] = df['IsActiveMember'] * df['NumOfProducts']

    return df