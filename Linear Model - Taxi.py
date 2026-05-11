#importing libraries
#data
import numpy as np
import pandas as pd
#machine learning
import keras
import ml_edu.experiment
import ml_edu.results
#data visualization
import plotly.express as px

#loading dataset - chicago taxi fare prediction from variables such as trip miles, etc..
importeddataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
datatable = importdataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

#Exploratory data analysis of the 6 variables
print('Number of observations: {0}\n\n'.format(len(datatable.index)))
datatable.describe()
null_count = datatable.isnull().sum().sum()
nans_count = datatable.isna().sum().sum()
print(f"Total number of null values: {null_count}")
print(f"Total number of null values: {nans_count}")

#Generating Correlation Matrix
datatable.corr(numeric_only = True) #corr close to 0: TIP_RATE with the three other numerical variables
px.scatter_matrix(datatable, dimensions=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
### Will Drop TIP_RATE for our model, has least correlation with other variables.

#Defining functions to build and train models
def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
  """Create and compile a simple linear regression model."""
  # Describe the topography of the model.
  # The topography of a simple linear regression model
  # is a single node in a single layer.
  inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
  concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
  outputs = keras.layers.Dense(units=1)(concatenated_inputs)
  model = keras.Model(inputs=inputs, outputs=outputs)

  # Compile the model topography into code that Keras can efficiently
  # execute. Configure training to minimize the model's mean squared error.
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                loss="mean_squared_error",
                metrics=metrics)

  return model

def train_model(
    trialname: str,
    labelname: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
  """Train the model by feeding it data."""

  # Feed the model the feature and the label.
  # The model will train for the specified number of epochs.
  features = {name: dataset[name].values for name in settings.input_features}
  label = dataset[labelname].values
  history = model.fit(x=features,
                      y=label,
                      batch_size=settings.batch_size,
                      epochs=settings.number_epochs)

  return ml_edu.experiment.Experiment(
      name=trialname,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )

# Model Trial/Fitting
datatable['TRIP_MINUTES'] = datatable['TRIP_SECONDS']/60
settings_1 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.002,
    number_epochs = 20,
    batch_size = 500,
    input_features = ['TRIP_MILES', 'TRIP_MINUTES']
)
metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]
model_1 = create_model(settings_1, metrics)
experiment_1 = train_model('two_features', model_1, datatable, 'FARE', settings_1)
ml_edu.results.plot_experiment_metrics(experiment_1, ['rmse'])
ml_edu.results.plot_model_predictions(experiment_1, datatable, 'FARE')

# Define functions to make predictions
def format_currency(x):
  return "${:.2f}".format(x)

def build_batch(df, batch_size):
  batch = df.sample(n=batch_size).copy()
  batch.set_index(np.arange(batch_size), inplace=True)
  return batch

def predict_fare(model, df, features, label, batch_size=500):
  batch = build_batch(df, batch_size)
  pred_values = model.predict_on_batch(x={name: batch[name].values for name in features})

  data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [],
          features[0]: [], features[1]: []}
  for i in range(batch_size):
    predicted = pred_values[i][0]
    observed = batch.at[i, label]
    data["PREDICTED_FARE"].append(format_currency(predicted))
    data["OBSERVED_FARE"].append(format_currency(observed))
    data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
    data[features[0]].append(batch.at[i, features[0]])
    data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))

  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return

# Predict Fare
output = predict_fare(experiment_1.model,datatable, experiment_1.settings.input_features, 'FARE')
show_predictions(output)
