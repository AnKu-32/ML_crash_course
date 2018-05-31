#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 09:43:40 2018

@author: satyendra.kumar9175
"""
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics
import math

city_names = ["pune", "mumbai", "kolkata", "chennai"]
populations = ["2m", "4m", "3m", "2m"]

df = pd.DataFrame({"City Name": city_names, "Population": populations})
df = df.reindex(np.random.permutation(df.index))

df_features = df[["City Name"]]

df_targets = pd.DataFrame()
df_targets["Population"] = df[["Population"]]

x = np.random.randn(200000000)
y = np.random.permutation()
n_bins = 10000

plt.hist(x, bins=n_bins)

california_housing_dataframe = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
        np.random.permutation(california_housing_dataframe.index))

california_housing_dataframe.info()

features_taken = ["longitude",
                  "latitude",
                  "housing_median_age",
                  "total_rooms",
                  "total_bedrooms",
                  "population",
                  "households",
                  "median_income"]

features = california_housing_dataframe[features_taken]

targets = (california_housing_dataframe["median_house_value"] / 1000)

training_examples = features.head(12000)
training_targets = targets.head(12000)

validation_examples = features.tail(5000)
validation_targets = targets.tail(5000)


display.display(training_examples.describe())


def construct_feature_columns(input_features) :
    return  set(tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features)
    

construct_feature_columns(features_taken)

dict_of_features = dict(features).items()

def my_input_function(features,
                      targets,
                      batch_size=1,
                      shuffle=True,
                      num_epochs=None, ):
    features = {key:np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    if shuffle:
        ds = ds.shuffle(10000)
    
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


predict_validation_input_fn = lambda: my_input_function(validation_examples,
                                                            validation_targets,
                                                            num_epochs=1,
                                                            shuffle=False)

def train_nn_regression_model(
        learning_rate,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    
    periods = 10
    steps_per_period = steps / periods
    
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
            feature_columns=construct_feature_columns(training_examples),
            hidden_units=hidden_units,
            optimizer=my_optimizer)
    
    training_input_fn = lambda: my_input_function(training_examples,
                                                  training_targets,
                                                  batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_function(training_examples,
                                                          training_targets,
                                                          num_epochs=1,
                                                          shuffle=False)
    predict_validation_input_fn = lambda: my_input_function(validation_examples,
                                                            validation_targets,
                                                            num_epochs=1,
                                                            shuffle=False)
    
    print "Training model..."
    print "RMSE on training data: "
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        dnn_regressor.train(input_fn = training_input_fn, 
                            steps=steps_per_period)
        
        training_predictions = dnn_regressor.predict(input_fn = predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        
        validation_predictions = dnn_regressor.predict(input_fn = predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
        
        training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(training_targets, training_predictions))
        validation_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(validation_targets, validation_predictions))
        
        print "  period %02d : %0.2f" % (period, training_root_mean_squared_error)
        
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
        
    print "Model training finished"
        
    plt.xlabel("RMSE")
    plt.ylabel("Periods")
    plt.title("Root Mean Sqaured Error vs Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
        
    print "Final RMSE (on training data): %0.2f" %training_root_mean_squared_error
    print "Final RMSE (on validation data): %0.2f" %validation_root_mean_squared_error
        
    return dnn_regressor
        
        
dnn_regressor = train_nn_regression_model(
    learning_rate=0.001,
    steps=2000,
    batch_size=100,
    hidden_units=[10, 10],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)


california_housing_test_data = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_test.csv", sep=",")

test_examples = california_housing_test_data[features_taken]
test_targets = california_housing_test_data["median_house_value"] / 1000

predict_test_input_fn = lambda: my_input_function(test_examples, test_targets, num_epochs=1, shuffle=False)

test_predictions = dnn_regressor.predict(input_fn = predict_test_input_fn)
test_predictions = np.array([item['predictions'][0] for item in test_predictions])

root_mean_squared_error = math.sqrt(metrics.mean_squared_error(test_targets, test_predictions))

print "Final RMSE (on test data): %0.2f" % root_mean_squared_error












