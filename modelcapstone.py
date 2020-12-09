import time,os,re,csv,sys,uuid,joblib
from datetime import date
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from statsmodels.tsa.arima.model import ARIMA

from logger import update_predict_log, update_train_log
from cslib import fetch_ts, engineer_features

## model specific variables (iterate the version and note with each change)
MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learing model for time-series"

def _model_train(df,tag,test=False):
    """
    example funtion to train model
    
    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file 

    """


    ## start timer for runtime
    time_start = time.time()
    
    X,y,dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        y=y[mask]
        X=X[mask]
        dates=dates[mask]
        
    ## Create the model
    arma_mod300 = ARIMA(y,X, order=(3, 0, 0)).fit()
    y_pred = arma_mod300.predict()
    eval_rmse =  round(np.sqrt(mean_squared_error(y,y_pred)))
    
    ## retrain using all data
    model_name = re.sub("\.","_",str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(MODEL_DIR,
                                   "test-{}-{}.joblib".format(tag,model_name))
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join(MODEL_DIR,
                                   "sl-{}-{}.joblib".format(tag,model_name))
        print("... saving model: {}".format(saved_model))
        
    joblib.dump(arma_mod300,saved_model)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update log
#    (data_shape, eval_test, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=False):
    update_train_log((str(dates[0]),str(dates[-1])),{'rmse':eval_rmse},runtime, MODEL_VERSION, MODEL_VERSION_NOTE,test=True)
  

def model_train(data_dir=None,test=False):
    """
    funtion to train model given a df
    
    'mode' -  can be used to subset data essentially simulating a train
    """
    if not data_dir:
        data_dir = "cs-train"
    
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subseting data")
        print("...... subseting countries")
        
    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    ## train a different model for each data sets
    for country,df in ts_data.items():
        
        if test and country not in ['all','united_kingdom']:
            continue
        
        _model_train(df,country,test=test)
    
def model_load(data_dir=None,training=True, test=False):
    """
    example funtion to load model
    
    The prefix allows the loading of different models
    """
    prefix='sl'
    if test:
        prefix='test'

    if not data_dir:
        data_dir = "cs-train"
    
    models = [f for f in os.listdir(os.path.join(".","models")) if re.search(prefix,f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found did you train?".format(prefix))

    all_models = {}
    for model in models:
        all_models[re.split("-",model)[1]] = joblib.load(os.path.join(".","models",model))

    ## load data
    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country, df in ts_data.items():
        X,y,dates = engineer_features(df,training=training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {"X":X,"y":y,"dates": dates}
        
    return(all_data, all_models)

def model_predict(country,year,month,day,all_models=None,test=False):
    """
    example funtion to predict from model
    """

    ## start timer for runtime
    time_start = time.time()

    ## load model if needed
    if not all_models:
        all_data,all_models = model_load(training=False,test=test)
    
    ## input checks
    if country not in all_models.keys():
        raise Exception("ERROR (model_predict) - model for country '{}' could not be found".format(country))

    for d in [year,month,day]:
        if re.search("\D",d):
            raise Exception("ERROR (model_predict) - invalid year, month or day")
    
    ## load data
    model = all_models[country]
    data = all_data[country]

    ## check date
    target_date = "{}-{}-{}".format(year,str(month).zfill(2),str(day).zfill(2))
    print("Target Date:")
    print(target_date)

    if target_date not in data['dates']:
        raise Exception("ERROR (model_predict) - date {} not in range {}-{}".format(target_date,
                                                                                    data['dates'][0],
                                                                                    data['dates'][-1]))
    date_indx = np.where(data['dates'] == target_date)[0][0]
    print("Date Index")
    print(date_indx)
    #query = data['dates'].iloc[[date_indx]]
    
    ## make prediction and gather data for log entry
    y_pred = np.asscalar(model.predict(date_indx,date_indx).to_numpy(dtype='float64'))
    y_proba = None
    if 'predict_proba' in dir(model) and 'probability' in dir(model):
        if model.probability == True:
            y_proba = model.predict_proba(query)


    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update predict log
#    update_predict_log(y_pred, y_proba, query, runtime, MODEL_VERSION, test=False)
    update_predict_log(y_pred,y_proba,target_date,runtime, MODEL_VERSION, test=test)
    
    return({'y_pred':y_pred,'y_proba':y_proba})

if __name__ == "__main__":

    """
    basic test procedure for model.py
    """

    ## train the model
    print("TRAINING MODELS")
    data_dir = os.path.join("..","data","cs-train")
    model_train(data_dir,test=True)

    ## load the model
    print("LOADING MODELS")
    all_data, all_models = model_load()
    print("... models loaded: ",",".join(all_models.keys()))

    ## test predict
    country='all'
    year='2018'
    month='01'
    day='05'
    result = model_predict(country,year,month,day)
    print(result)