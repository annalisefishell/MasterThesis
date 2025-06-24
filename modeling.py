import plotting as p

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, root_mean_squared_error
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, MaxPool2D, Conv2D, Conv2DTranspose, \
   Concatenate, Dropout, LeakyReLU, Dense, Activation, MaxPooling2D, UpSampling2D
import time

def gbt(X_train, y_train, X_test, y_test, X_2d, height, width, 
       num_trees=50, max_tree_depth=15, output=True):
    # Fit linear regression model
    start = time.time()
    model = xgb.XGBRegressor(n_estimators=num_trees,
                             max_depth=max_tree_depth,
                             tree_method='hist'
                             ) 
    model.fit(X_train, y_train)

    # Predict
    pred = model.predict(X_test)

    # Compute metrics
    running_time = time.time() - start
    rmse = root_mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
   
    if output:
      print(f"Training Time: {running_time:.2f} seconds")
      print(f"RMSE: {rmse} Mg/ha")
      print(f"R² Score: {r2}")
      
      
      # Reshape prediction to original 2D format for visualization or further use
      pred_image = model.predict(X_2d).reshape(height, width)
      p.plot_raster(pred_image, 'MLR Prediction', 'rainbow', normalized=False, cbar_label='Mg/ha')
    
    return pred, running_time, rmse, r2


def rf(X_train, y_train, X_test, y_test, X_2d, height, width, 
       num_trees=50, max_tree_depth=15, output=True):
    # Fit linear regression model
    start = time.time()
    model = RandomForestRegressor(
        n_estimators=num_trees,       
        max_depth=max_tree_depth,          
        max_samples=0.25,      
        max_features='log2',        
        n_jobs=-1,              
        verbose=1
    )
    model.fit(X_train, y_train)

    # Predict
    pred = model.predict(X_test)

    # Compute metrics
    running_time = time.time() - start
    rmse = root_mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    if output:
        print(f"Training Time: {running_time:.2f} seconds")
        print(f"RMSE: {rmse} Mg/ha")
        print(f"R² Score: {r2}")
        
        # Reshape prediction to original 2D format for visualization or further use
        pred_image = model.predict(X_2d).reshape(height, width)
        p.plot_raster(pred_image, 'MLR Prediction', 'rainbow', normalized=False, cbar_label='Mg/ha')
    
    return pred, running_time, rmse, r2


def mlr(X_train, y_train, X_test, y_test, X_2d, height, width, output=True):
    # Fit linear regression model
    start = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    pred = model.predict(X_test)

    # Compute metrics
    running_time = time.time() - start
    rmse = root_mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)


    if output:
        print(f"Training Time: {running_time:.2f} seconds")
        print(f"RMSE: {rmse} Mg/ha")
        print(f"R² Score: {r2}")
        
        # Reshape prediction to original 2D format for visualization or further use
        pred_image = model.predict(X_2d).reshape(height, width)
        p.plot_raster(pred_image, 'MLR Prediction', 'rainbow', normalized=False, cbar_label='Mg/ha')
    
    return pred, running_time, rmse, r2

def parameter_experiments(num_trees, max_tree_depths, num_sims, model, X_train,
                          y_train, X_test, y_test, X_2d, height, width):
    output_dict = {}
    for tree_num in num_trees:
        for depth in max_tree_depths:
            time_list = []
            rmse_list = []
            r2_list = []
            for i in range(num_sims):
                if model == 'rf':
                    _, runtime, rmse, r2 = rf(X_train, y_train, X_test, y_test, X_2d, height, width, 
                                            output=False, num_trees=tree_num, max_tree_depth=depth)
                elif model == 'gbt':
                    _, runtime, rmse, r2 = gbt(X_train, y_train, X_test, y_test, X_2d, height, width, 
                                            output=False, num_trees=tree_num, max_tree_depth=depth)
                time_list.append(runtime)
                rmse_list.append(rmse)
                r2_list.append(r2)
            output_dict.update({str(tree_num) + ',' + str(depth): [time_list, rmse_list, r2_list]})
    return output_dict