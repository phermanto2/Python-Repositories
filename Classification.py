# Load Libraries Needed
import keras
import ml_edu.experiment
import ml_edu.results
import numpy as np
import pandas as pd
import plotly.express as px

# Load Dataset - Rice Classification
importeddataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")
ricedata = importeddataset[[
    'Area',
    'Perimeter',
    'Major_Axis_Length',
    'Minor_Axis_Length',
    'Eccentricity',
    'Convex_Area',
    'Extent',
    'Class',
]]

ricedata.describe()

# Create five 2D plots of the features against each other, color-coded by class.
for x_axis, y_axis in [
    ('Area', 'Eccentricity'),
    ('Convex_Area', 'Perimeter'),
    ('Major_Axis_Length', 'Minor_Axis_Length'),
    ('Perimeter', 'Extent'),
    ('Eccentricity', 'Major_Axis_Length'),
]:
  px.scatter(ricedata, x=x_axis, y=y_axis, color='Class').show()

# Create 3d Plots to assess features relationships again
# Plot major axis length, area and eccentricity, with observations
# color-coded by class.
px.scatter_3d(
    ricedata,
    x='Eccentricity',
    y='Area',
    z='Major_Axis_Length',
    color='Class',
).show()


# When creating a model with multiple features, the values of each feature should 
# span roughly the same range. If one feature's values range from 500 to 100,000 
# and another feature's values range from 2 to 12, the model will need to have 
# weights of extremely low or extremely high values to be able to combine these 
# features effectively. This could result in a low quality model. 
# To avoid this, we should normalize the data
# Calculate the Z-scores of each numerical column in the raw data and write
# them into a new DataFrame named df_norm.
feature_means = ricedata.mean(numeric_only=True)
feature_stdev = ricedata.std(numeric_only=True)
num_features = ricedata.select_dtypes('number').columns
normalized_data = (
    ricedata[num_features] - feature_means
) / feature_stdev
# Attach the (copy of) class to the new dataframe
normalized_data['Class'] = ricedata['Class']

# Set seed
keras.utils.set_random_seed(42)

# Create a column setting the Cammeo label to '1' and the Osmancik label to '0'
normalized_data['Class_Bool'] = (
    # Returns true if class is Cammeo, and false if class is Osmancik
    normalized_data['Class'] == 'Cammeo'
).astype(int)


# Create indices at the 80th and 90th percentiles
num_samples = len(normalized_data)
index_80th = round(num_samples * 0.8)
index_90th = index_80th + round(num_samples * 0.1)

# Randomize order and split into train, validation, and test with a 80%, 10%, 10% split
shuffled_dataset = normalized_data.sample(frac=1, random_state=100)
train_data = shuffled_dataset.iloc[0:index_80th]
validation_data = shuffled_dataset.iloc[index_80th:index_90th]
test_data = shuffled_dataset.iloc[index_90th:]

# Prevent the model from getting the label as input during training (a.k.a label leakage).
# This can be done by storing features and labels as separate variables
label_columns = ['Class', 'Class_Bool']
train_features = train_data.drop(columns=label_columns)
train_labels = train_data['Class_Bool'].to_numpy()
validation_features = validation_data.drop(columns=label_columns)
validation_labels = validation_data['Class_Bool'].to_numpy()
test_features = test_data.drop(columns=label_columns)
test_labels = test_data['Class_Bool'].to_numpy()

############# TRAINING MODEL
# Listing our features for training the model
all_input_features = [
  'Eccentricity',
  'Major_Axis_Length',
  'Minor_Axis_Length',
  'Area',
  'Convex_Area',
  'Perimeter',
  'Extent',
]

# Define the functions that create and train a model.
def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
  """Create and compile a simple classification model."""
  model_inputs = [
      keras.Input(name=feature, shape=(1,))
      for feature in settings.input_features
  ]
  # Use a Concatenate layer to assemble the different inputs into a single
  # tensor which will be given as input to the Dense layer.
  concatenated_inputs = keras.layers.Concatenate()(model_inputs)
  model_output = keras.layers.Dense(
      units=1, name='dense_layer', activation=keras.activations.sigmoid
  )(concatenated_inputs)
  model = keras.Model(inputs=model_inputs, outputs=model_output)
  # Call the compile method to transform the layers into a model that
  # Keras can execute.  Notice that we're using a different loss
  # function for classification than for regression.
  model.compile(
      optimizer=keras.optimizers.RMSprop(
          settings.learning_rate
      ),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics,
  )
  return model


def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    labels: np.ndarray,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
  """Feed a dataset into the model in order to train it."""
  # The x parameter of keras.Model.fit can be a list of arrays, where
  # each array contains the data for one feature.
  features = {
      feature_name: np.array(dataset[feature_name])
      for feature_name in settings.input_features
  }

  history = model.fit(
      x=features,
      y=labels,
      batch_size=settings.batch_size,
      epochs=settings.number_epochs,
  )
  return ml_edu.experiment.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )
  
######### TRAINING ALL FEATURES, CLASSIFICATION THRESHOLD AT 0.5
settings_func = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.5,
    input_features=all_input_features,
)

# Modify the following definition of METRICS to generate
# not only accuracy and precision, but also recall:
metrics = [
    keras.metrics.BinaryAccuracy(
        name='accuracy',
        threshold=settings_func.classification_threshold,
    ),
    keras.metrics.Precision(
        name='precision',
        thresholds=settings_func.classification_threshold,
    ),
    keras.metrics.Recall(
        name='recall', thresholds=settings_func.classification_threshold
    ),
    keras.metrics.AUC(num_thresholds=100, name='auc'),
]

# Establish the model's topography.
model_all_features = create_model(settings_func, metrics)

# Train the model on the training set.
allfeatures_model = train_model(
    'all features',
    model_all_features,
    train_features,
    train_labels,
    settings_func,
)

# Plot metrics vs. epochs
ml_edu.results.plot_experiment_metrics(
    allfeatures_model, ['accuracy', 'precision', 'recall']
)
ml_edu.results.plot_experiment_metrics(allfeatures_model, ['auc'])

# Evaluate full-featured model on validation split
validation_metrics = allfeatures_model.evaluate(
    validation_features,
    validation_labels,
)
compare_train_validation(allfeatures_model, validation_metrics)

# Computing Final Test Metrics
test_performance_metrics = allfeatures_model.evaluate(
    test_features,
    test_labels,
)
for metric, test_value in test_performance_metrics.items():
  print(f'Test {metric}:  {test_value:.4f}')
