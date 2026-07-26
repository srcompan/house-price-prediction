import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,r2_score
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
matplotlib.rcParams["figure.figsize"] = (20,10)
import joblib

import pandas as pd

# Replace 'your_file.csv' with the actual filename
df1 = pd.read_csv('bhv.csv')
df1.head()

df2 = df1.drop(['availability','area_type'],axis='columns')
df2.shape

df3 = df2.dropna()
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


df3[~df3['total_sqft'].apply(is_float)].head(10)
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    
df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]


df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
location_stats

location_stats_less_than_10 = location_stats[location_stats<=10]

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

df6 = df5[~(df5.total_sqft/df5.bhk<300)]

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)

df9 = df8[df8.bath<df8.bhk+2]

df10 = df9.drop(['size','price_per_sqft'],axis='columns')

label_encoder_location = LabelEncoder() 
label_encoder_society = LabelEncoder()

# Fit and transform the data
df10['location'] = label_encoder_location.fit_transform(df10['location'])
df10['society'] = label_encoder_society .fit_transform(df10['society'])

joblib.dump(label_encoder_location, 'label_encoder_location.pkl')
joblib.dump(label_encoder_society, 'label_encoder_society.pkl')


X = df10.drop(['price'],axis='columns')
print(X.columns)
y = df10.price

from sklearn.model_selection import GridSearchCV, train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2   ,random_state=42)

# from sklearn.linear_model import Lasso, LinearRegression
# lr_clf = LinearRegression()
# lr_clf.fit(X_train,y_train)
# ypred=lr_clf.predict(X_test)
# print(r2_score(y_test, ypred))

poly = PolynomialFeatures(degree=2)  # You can experiment with the degree (e.g., 2, 3, or higher)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Step 2: Train a linear regression model on the transformed polynomial features
poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train)

# Step 3: Make predictions
ypred_poly = poly_lr.predict(X_test_poly)
r2_poly = r2_score(y_test, ypred_poly)

print(f'R² (Coefficient of Determination): {r2_poly}')

dt_regressor = GradientBoostingRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Make predictions on the test set
ypred_dt = dt_regressor.predict(X_test)
r2_dt = r2_score(y_test, ypred_dt)
print(f'R² (Coefficient of Determination): {r2_dt}')


param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'subsample': [0.8, 0.9, 1.0]
}

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Get the best parameters
print(f'Best Parameters: {grid_search.best_params_}')

# Train the model with the best parameters
best_gbr = grid_search.best_estimator_

# Make predictions with the tuned model
ypred_best_gbr = best_gbr.predict(X_test)

# Evaluate the tuned model

r2_best_gbr = r2_score(y_test, ypred_best_gbr)

# Print the evaluation metrics for the tuned model
print(f'R² (Coefficient of Determination): {r2_best_gbr}')
joblib.dump(best_gbr, 'best_gbr_model.pkl')