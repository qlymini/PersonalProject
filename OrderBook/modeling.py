import re
import os
import glob
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from matplotlib import cm, pyplot as plt
from sklearn.linear_model import Lasso
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

PATH = "/Users/liyu/PycharmProjects/pythonProject/3red_test/results_csv"
CSV_DATA = glob.glob(os.path.join(PATH, "*.csv"))
Target = 'microprice'
agg_df = pd.DataFrame()

def read_csv_data(file):
    date = re.search("([0-9]{8})", file)
    date = pd.to_datetime(date.group(), format='%Y%m%d')
    df = pd.read_csv(file)
    df['date'] = date
    return df

def diffvol_ploting(source_data):
    diff_volumes = source_data.aq0 - source_data.bq0

    plt.hist(diff_volumes[~np.isnan(diff_volumes)], bins = 30)
    plt.title('Difference of ask-bid volumes')
    plt.xlabel('Volume difference')
    plt.ylabel('Dist')

    plt.show()

def midprice_plotting(source_data):
    spread = source_data.ap0 - source_data.bp0
    spread = spread.iloc[1:,]
    plt.hist(spread[~np.isnan(spread)], bins=30)
    plt.title('Difference of spread')
    plt.xlabel('mid price difference')
    plt.ylabel('Dist')

    plt.show()

def relation_plotting(df):

    microprice = np.log((df['bq0']*df['ap0']+df['aq0']*df['bp0'])/(df['bq0']+df['aq0'])+1)
    spread0 = np.log((df.ap0 - df.bp0)+1)
    diff_volumes0 = np.log(df.aq0 - df.bq0+1)

    plt.subplot(1,2,1)
    plt.plot(spread0,microprice,'ro')
    plt.title('spread0 vs microprice0')

    plt.subplot(1, 2, 2)
    plt.plot(diff_volumes0, microprice, 'ro')
    plt.title('diff_volumes0 vs microprice0')

    plt.show()

def orderbook_dataprocess(data, level, num_lag):
    df = data.copy()
    transformed_df = pd.DataFrame()
    df = df[~(df.ap0.isna() &df.bp0.isna())]

    for i in range(0,  level):
        transformed_df['spread{0}'.format(i)] = np.log(df['ap{0}'.format(i)]-df['bp{0}'.format(i)]+1)
        transformed_df['voldiff{0}'.format(i)] = np.log(df['aq{0}'.format(i)] - df['bq{0}'.format(i)] + 1)
        #transformed_df['ask_vol_change{0}'.format(i)] = np.log(df['aq{0}'.format(i)].pct_change()+1)
        #transformed_df['bid_vol_change{0}'.format(i)] = np.log(df['bq{0}'.format(i)].pct_change()+1)

    transformed_df['microprice'] = np.log((df['bq0']*df['ap0']+df['aq0']*df['bp0'])/(df['bq0']+df['aq0'])+1)

    transformed_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return transformed_df.dropna()

def correlation(df, threshold):
    col_corr = set()
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


def feature_selection_forward(transformed_df):
    X_train, X_test, y_train, y_test = train_test_split(
        transformed_df.drop(labels=[Target], axis=1),
        transformed_df[Target],
        test_size=0.3,
        random_state=1)
    corr_features = correlation(X_train, 0.6)
    X_train.drop(labels=corr_features, axis=1, inplace=True)
    X_test.drop(labels=corr_features, axis=1, inplace=True)
    X_train.fillna(0, inplace=True)


    sfs1 = SFS(RandomForestRegressor(),
               k_features=2,
               forward=True,
               floating=False,
               verbose=2,
               scoring='r2',
               cv=3)

    sfs1 = sfs1.fit(np.array(X_train), y_train)
    X_train = X_train[list(X_train.columns[list(sfs1.k_feature_idx_)])]
    X_test = X_test[list(X_test.columns[list(sfs1.k_feature_idx_)])]
    print(list(sfs1.k_feature_idx_))
    return X_train, X_test, y_train, y_test

def recursive_features_elimate(transformed_df):
    X_train, X_test, y_train, y_test = train_test_split(
        transformed_df.drop(labels=[Target], axis=1),
        transformed_df[Target],
        test_size=0.3,
        random_state=0)
    corr_features = correlation(X_train, 0.5)
    X_train.drop(labels=corr_features, axis=1, inplace=True)
    X_test.drop(labels=corr_features, axis=1, inplace=True)
    X_train.fillna(0, inplace=True)
    # Backward Elimination
    cols = list(X_train.columns)
    while (len(cols) > 0):
        X_1 = X_train[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y_train, X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index=cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if (pmax > 0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    X_train = X_train[selected_features_BE]
    X_test=X_test[selected_features_BE]
    print(selected_features_BE)
    return X_train,X_test,y_train,y_test

def feature_without_selection(transformed_df):
    X_train, X_test, y_train, y_test = train_test_split(
            transformed_df.drop(labels=[Target], axis=1),
            transformed_df[Target],
            test_size=0.3,
            random_state=0)
    corr_features = correlation(X_train, 0.5)
    X_train.drop(labels=corr_features, axis=1, inplace=True)
    X_test.drop(labels=corr_features, axis=1, inplace=True)
    X_train.fillna(0, inplace=True)

    return X_train, X_test, y_train, y_test

def start_prediction(X_train, X_test, y_train, y_test,data_set):
    actual_rs = list(data_set[Target])
    actual_rs = np.asarray(actual_rs)



    lasso_reg = Lasso(normalize=False)


    lasso_reg.fit(X_train, y_train)

    y_pred_lass = lasso_reg.predict(X_test)
    r2 = np.sqrt(mean_squared_error(y_test,y_pred_lass))
    print("\n\nLasso R Square : ", r2)

    return r2, lasso_reg

def lasso_tunning(X_train, X_test, y_train,y_test):
    lasso_cv_model = LassoCV(alphas=np.random.randint(50, 1000, 100), cv=10, max_iter=100000).fit(X_train, y_train)
    lasso_tuned = Lasso().set_params(alpha=lasso_cv_model.alpha_).fit(X_train, y_train)

    y_pred_tuned = lasso_tuned.predict(X_test)

    r_sqr=np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    print(r_sqr)
    return r_sqr, lasso_tuned


if __name__=="__main__":
    df = read_csv_data(CSV_DATA[0])
    relation_plotting(df)
    processed_data = orderbook_dataprocess(df, 5, 1)
    X_train, X_test, y_train, y_test =recursive_features_elimate(processed_data)
    score_rs , lasso_reg = start_prediction(X_train, X_test, y_train,y_test, processed_data)
    print(lasso_reg.coef_)
    r2, lasso_tuned = lasso_tunning(X_train, X_test, y_train,y_test)
    print(lasso_tuned.coef_)


