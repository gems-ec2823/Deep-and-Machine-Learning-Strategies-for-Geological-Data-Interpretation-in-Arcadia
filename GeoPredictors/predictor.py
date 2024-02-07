import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import r2_score

def load_and_preprocess_data():
    """
    Load and preprocess the data.

    Returns:
        pd.DataFrame: The preprocessed data.
    """

    
    def replace_special_characters(value):
        """
        Replace special characters in a string.

        Args:
            value (str): The input string.

        Returns:
            str: The string with special characters replaced.
        """
        value = str(value)
        result = ''
        i = 0

        while i < len(value):
            # substitute special characters
            match = re.match(r'[0-9.,<>]+', value[i:])
            if match:
                matched_part = match.group()
                result += matched_part
                i += len(matched_part)
            else:
                # if no match, append the character and move on
                i += 1

        # substitute the comma with a dot
        result = result.replace(',', '.')
        # substitute the less than sign with a zero, as it is too small
        result = result.replace('< .01', '0.001').replace('<.01', '0.001')

        return result
    
    def add_well_data_to_df(df, W):
        """
        Add well data to a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to add the well data to.
            W (pd.DataFrame): The well data DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with the added well data.
        """
        # Get the list of columns from W excluding 'DEPTH'
        well_columns = [col for col in W.columns if col != 'DEPTH']
        
        # Add new columns to df and initialize with NaN
        for col in well_columns:
            df[col] = np.nan

        # Iterate over each row in df
        for idx, row in df.iterrows():
            # Find the row in W with the closest DEPTH
            closest_idx = np.abs(W['DEPTH'] - row['DEPTH']).idxmin()
            closest_row = W.iloc[closest_idx]

            # Update the relevant columns in df
            for col in well_columns:
                df.at[idx, col] = closest_row[col]

        return df
    
    # DATA LOADING
    df1 = pd.read_csv('Data/labels/permeability/204_20_1.csv')
    df1Z = pd.read_csv('Data/labels/permeability/204_20_1Z.csv')
    df2 = pd.read_csv('Data/labels/permeability/204_20_2.csv')
    df3 = pd.read_csv('Data/labels/permeability/204_20_3.csv')
    df6 = pd.read_csv('Data/labels/permeability/204_24a_6.csv')
    df6A = pd.read_csv('Data/labels/permeability/204_20_6A.csv')
    df7 = pd.read_csv('Data/labels/permeability/204_19_7.csv')
    df7_20A = pd.read_csv('Data/labels/permeability/204_20a_7.csv')
    df7_24A = pd.read_csv('Data/labels/permeability/204_24a_7.csv')

    W19_3 = pd.read_csv('csv_wireline_logs/well_204_19_3A.csv')
    W19_6 = pd.read_csv('csv_wireline_logs/well_204_19_6.csv')
    W19_7 = pd.read_csv('csv_wireline_logs/well_204_19_7.csv')
    W1 = pd.read_csv('csv_wireline_logs/well_204_20_1.csv')
    W1Z = pd.read_csv('csv_wireline_logs/well_204_20_1Z.csv')
    W2 = pd.read_csv('csv_wireline_logs/well_204_20_2.csv')
    W3 = pd.read_csv('csv_wireline_logs/well_204_20_3.csv')
    W6a = pd.read_csv('csv_wireline_logs/well_204_20_6a.csv')
    W6 = pd.read_csv('csv_wireline_logs/well_204_24a_6.csv')
    W7 = pd.read_csv('csv_wireline_logs/well_204_20a_7.csv')
    W7_24A = pd.read_csv('csv_wireline_logs/well_204_24a_7.csv')

    # DATA PREPROCESSING
    df3['DEPTH'] = df3['DEPTH'].apply(replace_special_characters)
    df3['DEPTH'] = df3['DEPTH'].astype(float)

    df6A['DEPTH'] = df6A['DEPTH'].apply(replace_special_characters)
    df6A['DEPTH'] = df6A['DEPTH'].astype(float)

    df7_20A['DEPTH'] = df7_20A['DEPTH'].apply(replace_special_characters)
    df7_20A['DEPTH'] = df7_20A['DEPTH'].astype(float)

    df7_24A['DEPTH'] = df7_24A['DEPTH'].apply(replace_special_characters)
    df7_24A = df7_24A[df7_24A['DEPTH'] != '']
    df7_24A['DEPTH'] = df7_24A['DEPTH'].astype(float)

    df7['DEPTH'] = df7['DEPTH'].apply(replace_special_characters)
    df7 = df7[df7['DEPTH'] != '']
    df7['DEPTH'] = df7['DEPTH'].astype(float)

    # Providing the latitude and longitude of the wells
    
    df1['longitude'] = -4.0386472
    df1Z['longitude'] = -4.0386472
    df2['longitude'] = -4.0585925
    df3['longitude'] = -4.0429056
    df6['longitude'] = -4.3999889
    df6A['longitude'] = -4.0310681
    df7['longitude'] = -4.2432231
    df7_20A['longitude'] = -4.1381753
    df7_24A['longitude'] = -4.2365475

    df1['latitude'] = 60.3518389
    df1Z['latitude'] = 60.3518389
    df2['latitude'] = 60.3465144
    df3['latitude'] = 60.4264917
    df6['latitude'] = 60.323475
    df6A['latitude'] = 60.3958653
    df7['latitude'] = 60.3731917
    df7_20A['latitude'] = 60.3443039
    df7_24A['latitude'] = 60.3305383

    # ALIGNING THE WELL CORE DATA WITH THE WIRELINE LOG DATA
    ## df1 
    start_depth = 1945.00
    end_depth = 2116.00
    df1.loc[(df1['DEPTH'] >= start_depth) & (df1['DEPTH'] <= end_depth), 'DEPTH'] += 3.0

    ## df1Z
    start_depth = 2673.00
    end_depth = 2686.46
    df1Z.loc[(df1Z['DEPTH'] >= start_depth) & (df1Z['DEPTH'] <= end_depth), 'DEPTH'] += 5.0

    ## df2
    start_depth = 1998.00
    end_depth = 2018.55
    df2.loc[(df2['DEPTH'] >= start_depth) & (df2['DEPTH'] <= end_depth), 'DEPTH'] += 2.0

    ## df3
    start_depth = 2401.00
    end_depth = 2425.75
    df3.loc[(df3['DEPTH'] >= start_depth) & (df3['DEPTH'] <= end_depth), 'DEPTH'] -= 1.2
    start_depth2 = 2427.50
    end_depth2 = 2436.83
    df3.loc[(df3['DEPTH'] >= start_depth2) & (df3['DEPTH'] <= end_depth2), 'DEPTH'] -= 2.7
    start_depth3 = 2658.00
    end_depth3 = 2685.79
    df3.loc[(df3['DEPTH'] >= start_depth3) & (df3['DEPTH'] <= end_depth3), 'DEPTH'] += 1.2
    start_depth4 = 2958.00
    end_depth4 = 2978.15
    df3.loc[(df3['DEPTH'] >= start_depth4) & (df3['DEPTH'] <= end_depth4), 'DEPTH'] += 1.6

    ## df6
    df6['DEPTH'] = df6['DEPTH'].apply(replace_special_characters)
    df6 = df6[df6['DEPTH'] != '']
    df6['DEPTH'] = df6['DEPTH'].astype(float)
    start_depth = 2211.00
    end_depth = 2230.70
    df6.loc[(df6['DEPTH'] >= start_depth) & (df6['DEPTH'] <= end_depth), 'DEPTH'] += 3.5
    start_depth2 = 2416.00
    end_depth2 = 2429.67
    df6.loc[(df6['DEPTH'] >= start_depth2) & (df6['DEPTH'] <= end_depth2), 'DEPTH'] += 2.0
    start_depth3 = 2429.67
    end_depth3 = 2438.96
    df6.loc[(df6['DEPTH'] >= start_depth3) & (df6['DEPTH'] <= end_depth3), 'DEPTH'] += 2.43
    start_depth4 = 2440.20
    end_depth4 = 2462.93
    df6.loc[(df6['DEPTH'] >= start_depth4) & (df6['DEPTH'] <= end_depth4), 'DEPTH'] += 0.8
    start_depth5 = 2485.60
    end_depth5 = 2493.50
    df6.loc[(df6['DEPTH'] >= start_depth5) & (df6['DEPTH'] <= end_depth5), 'DEPTH'] += 0.4

    ## df7
    start_depth = 2131.00
    end_depth = 2157.68
    df7.loc[(df7['DEPTH'] >= start_depth) & (df7['DEPTH'] <= end_depth), 'DEPTH'] += 1.8
    start_depth2 = 2158.00
    end_depth2 = 2180.38
    df7.loc[(df7['DEPTH'] >= start_depth2) & (df7['DEPTH'] <= end_depth2), 'DEPTH'] += 1.6
    start_depth3 = 2540.00
    end_depth3 = 2544.89
    df7.loc[(df7['DEPTH'] >= start_depth3) & (df7['DEPTH'] <= end_depth3), 'DEPTH'] += 3.0

    # df7_24A
    start_depth = 2075.00
    end_depth = 2102.00
    df7_24A.loc[(df7_24A['DEPTH'] >= start_depth) & (df7_24A['DEPTH'] <= end_depth), 'DEPTH'] += 2.2
    start_depth2 = 2102.00
    end_depth2 = 2127.85
    df7_24A.loc[(df7_24A['DEPTH'] >= start_depth2) & (df7_24A['DEPTH'] <= end_depth2), 'DEPTH'] += 2.35
    start_depth3 = 2128.50
    end_depth3 = 2145.90
    df7_24A.loc[(df7_24A['DEPTH'] >= start_depth3) & (df7_24A['DEPTH'] <= end_depth3), 'DEPTH'] += 2.0
    start_depth4 = 2146.00
    end_depth4 = 2162.60
    df7_24A.loc[(df7_24A['DEPTH'] >= start_depth4) & (df7_24A['DEPTH'] <= end_depth4), 'DEPTH'] += 1.8
    start_depth5 = 2163.00
    end_depth5 = 2179.30
    df7_24A.loc[(df7_24A['DEPTH'] >= start_depth5) & (df7_24A['DEPTH'] <= end_depth5), 'DEPTH'] += 1.2
    start_depth6 = 2180.00
    end_depth6 = 2188.56
    df7_24A.loc[(df7_24A['DEPTH'] >= start_depth6) & (df7_24A['DEPTH'] <= end_depth6), 'DEPTH'] += 1.0
    start_depth7 = 2189.00
    end_depth7 = 2206.92
    df7_24A.loc[(df7_24A['DEPTH'] >= start_depth7) & (df7_24A['DEPTH'] <= end_depth7), 'DEPTH'] += 0.85
    start_depth8 = 2208.50
    end_depth8 = 2240.65
    df7_24A.loc[(df7_24A['DEPTH'] >= start_depth8) & (df7_24A['DEPTH'] <= end_depth8), 'DEPTH'] += 0.2

    df1_updated = add_well_data_to_df(df1, W1)
    df1Z_updated = add_well_data_to_df(df1Z, W1Z)
    df2_updated = add_well_data_to_df(df2, W2)
    df3_updated = add_well_data_to_df(df3, W3)
    df6_updated = add_well_data_to_df(df6, W6)
    df6A_updated = add_well_data_to_df(df6A, W6a)
    df7_updated = add_well_data_to_df(df7, W19_7)
    df7_20A_updated = add_well_data_to_df(df7_20A, W7)
    df7_24A_updated = add_well_data_to_df(df7_24A, W7_24A)

    df_final = pd.concat([df1_updated, df1Z_updated, df2_updated, df3_updated, df6_updated, df6A_updated, df7_updated, df7_20A_updated, df7_24A_updated], axis=0)
    df_final['POROSITY\n(HELIUM)'].replace('2B.3', '28.3', inplace = True)
    df_final = df_final.applymap(replace_special_characters)
    df_final = df_final.applymap(lambda x: float(x) if x else np.nan)
    
    # Lose the unnamed columns with no data in them from the dataset
    df_final = df_final.dropna(how='all', axis=1)
    df_final = df_final.dropna(subset=['PERMEABILITY (HORIZONTAL)\nKair\nmd'])

    #FEATURE ENGINEERING
    #Fix the density data
    df_final['DEN_SUM'] = df_final['DENC'] + df_final['DENS']
    df_final = df_final.drop(columns=['DENC', 'DENS'])

    #Drop features that we don't use
    df_final.drop(columns=['RMED', 'DTS1', 'DTS2', 'RACH', 'RLA2', 'RLA4', 'RPCH', 'DTS_1', 'DTS_2', 'AT20', 'AT60', 'DTS'], axis=1, inplace=True)
    #NB after exploratory data analysis we found that we did not have enough data in the above datasets to usefully lose the data.
    
    #Take logs of the selected features
    df_final['log_GR'] = np.log(df_final['GR'])
    df_final['log_RDEP'] = np.log(df_final['RDEP'])
    df_final['log_RMIC'] = np.log(df_final['RMIC'])
    df_final['log_RSHAL'] = np.log(df_final['RSHAL'])

    return df_final

def split_data(df):
    """
    Split the data into training, validation, and test sets.

    Parameters:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    X_train (pandas.DataFrame): The training features.
    X_val (pandas.DataFrame): The validation features.
    X_test (pandas.DataFrame): The test features.
    y_train (pandas.Series): The training target.
    y_val (pandas.Series): The validation target.
    y_test (pandas.Series): The test target.
    """
    features = ['DEPTH', 'POROSITY\n(HELIUM)', 'DEN_SUM', 'DTC', 'log_GR', 'PEF', 'log_RDEP', 'latitude']
    target = 'PERMEABILITY (HORIZONTAL)\nKair\nmd'

    # Dropping rows with NaN values in the specified columns
    df = df.dropna(subset=features + [target])

    # Splitting the dataset into training, validation, and test sets
    X = df[features]
    y = df[target]
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% training + validation, 20% test
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 60% training, 20% validation

    return X_train, X_val, X_test, y_train, y_val, y_test

def train(X_train, y_train):
    """
    Trains a stacked ensemble model using Random Forest and XGBoost regressors.

    Parameters:
    - X_train (array-like): The input features for training.
    - y_train (array-like): The target variable for training.

    Returns:
    - pipeline (Pipeline): The trained stacked ensemble model.
    """
    rf_model = RandomForestRegressor(max_depth=10, n_estimators=200, random_state=42)
    xgb_model = xgb.XGBRegressor(max_depth=6, n_estimators=100, objective='reg:squarederror', random_state=42)

    '''

    the hyperparameter grid for Random Forest and XGBoost regressors are done using GridSearchCV with 3-fold cross-validation.
    the parameters are as follows: 
    rf_param_grid = {
        'max_depth': [5, 10, 15],
        'n_estimators': [100, 200, 300]
    }

    xgb_param_grid = {
        'max_depth': [4, 6, 8],
        'n_estimators': [50, 100, 150]
    }
    
    '''

    stacked = StackingRegressor(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model)
        ],
        final_estimator=LinearRegression()
    )
    '''
    explain about the stackings here:
    
    StackingRegressor is a meta-estimator that fits a regressor on the whole datase
    while the final estimator is fitted using the predictions of the base estimators as features.
    
    '''

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('stacked', stacked)
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

def predict(model, X_test, y_test):
    """
    Predicts the target variable using the given model and returns the predicted values and R-squared score.

    Parameters:
    model (object): The trained model used for prediction.
    X_test (array-like): The input features for prediction.
    y_test (array-like): The true target values.

    Returns:
    tuple: A tuple containing the predicted values and R-squared score.
    """

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return y_pred, r2