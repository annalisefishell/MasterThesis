import plotting as p

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, root_mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPool2D, Conv2D, Conv2DTranspose, \
      Concatenate, Dropout, Dense, Activation, MaxPooling2D, \
      UpSampling2D, Flatten, BatchNormalization, Add, Resizing, concatenate
from tensorflow.keras.optimizers import Adam
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
      p.plot_raster(pred_image, 'GBT Prediction', 'BuGn', normalized=False, cbar_label='Mg/ha')
    
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
        p.plot_raster(pred_image, 'RF Prediction', 'BuGn', normalized=False, cbar_label='Mg/ha')
    
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
        p.plot_raster(pred_image, 'MLR Prediction', 'BuGn', normalized=False, cbar_label='Mg/ha')
    
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


def cnn_r2_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    true_mean = tf.reduce_mean(y_true)
    pred_mean = tf.reduce_mean(y_pred)

    cov = tf.reduce_sum((y_true - true_mean) * (y_pred - pred_mean))
    var_true = tf.reduce_sum(tf.square(y_true - true_mean))
    var_pred = tf.reduce_sum(tf.square(y_pred - pred_mean))
    corr = cov / (tf.sqrt(var_true * var_pred) + tf.keras.backend.epsilon())
    
    return tf.square(corr)


def cnn(X_train, y_train, X_test, y_test, original_shape=(0,0,0), output=False, patch_size=200, 
        architecture='simple'):
    start = time.time()
    input_shape = (patch_size, patch_size, X_train.shape[-1])

    if architecture=='simple':
        model = Sequential([
            Input(shape=input_shape),
            Conv2D(16, 3, padding='same', activation='relu'),
            MaxPooling2D(2),
            Conv2D(32, 3, padding='same', activation='relu'),
            UpSampling2D(2),  # restore spatial dims
            Conv2D(original_shape, 3, padding='same', activation='linear')
        ])

    if output:
        model.summary()

    model.compile(
        optimizer = 'adam',
        loss='mse',
        metrics=['root_mean_squared_error', cnn_r2_score]
    )

    # 6. Train using NumPy arrays
    model.fit(
        x=X_train,
        y=y_train,
        batch_size=16,
        validation_data=(X_test, y_test),
        epochs=10
    )
    
    runtime = time.time() - start
    loss, rmse, r2 = model.evaluate(X_test, y_test, batch_size=16)

    if output:
        print(f"Training Time: {runtime:.2f} seconds")
        print(f"Test MSE: {loss:.4f}")
        print(f"Test RMSE: {rmse}")
        print(f"Test R2: {r2}")

    return model, runtime, rmse, r2

def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def unet_encoder_block(inputs, num_filters, bottleneck=False): # work on documentation here down
   '''Builds the encoder sections of the unet models'''
  
   # Convolution with 3x3 filter followed by ReLU activation 
   x = Conv2D(num_filters, 3, padding = 'same')(inputs) 
   x = Activation('relu')(x) 
      
   # Convolution with 3x3 filter followed by ReLU activation 
   x = Conv2D(num_filters, 3, padding = 'same')(x) 
   x = Activation('relu')(x) 
  
   if not bottleneck:
      # Max Pooling with 2x2 filter 
      x = MaxPool2D(pool_size = (2, 2), strides = 2)(x) 
      
   return x

def unet_decoder_block(inputs, skip_features, num_filters): 
   '''Builds the decoder sections of the unet models'''

	# Upsampling with 2x2 filter
   x = Conv2DTranspose(num_filters, (2, 2), strides = 2, padding = 'same')(inputs) 
	
	# Copy and crop the skip features 
   skip_features = Resizing(x.shape[1], x.shape[2])(skip_features)
   x = Concatenate()([x, skip_features]) 
	
	# Convolution with 3x3 filter followed by ReLU activation 
   x = Conv2D(num_filters, 3, padding = 'same')(x) 
   x = Activation('relu')(x) 

	# Convolution with 3x3 filter followed by ReLU activation 
   x = Conv2D(num_filters, 3, padding = 'same')(x) 
   x = Activation('relu')(x) 
	
   return x

def inception_module(x, 
                     filters_1x1, 
                     filters_3x3_reduce, filters_3x3,
                     filters_5x5_reduce, filters_5x5,
                     filters_pool_proj, 
                     name=None):
    # 1x1 Convolution Branch
    branch1x1 = Conv2D(filters_1x1, (1,1), padding='same', activation='relu')(x)

    # 1x1 Convolution -> 3x3 Convolution Branch
    branch3x3 = Conv2D(filters_3x3_reduce, (1,1), padding='same', activation='relu')(x)
    branch3x3 = Conv2D(filters_3x3, (3,3), padding='same', activation='relu')(branch3x3)

    # 1x1 Convolution -> 5x5 Convolution Branch
    branch5x5 = Conv2D(filters_5x5_reduce, (1,1), padding='same', activation='relu')(x)
    branch5x5 = Conv2D(filters_5x5, (5,5), padding='same', activation='relu')(branch5x5)

    # MaxPooling -> 1x1 Convolution Branch
    branch_pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    branch_pool = Conv2D(filters_pool_proj, (1,1), padding='same', activation='relu')(branch_pool)

    # Concatenate all branches
    x = concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=3, name=name)
    return x