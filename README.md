# Arcadia project
## Applying Computational & Data Science: Team Poropermeables

## Project Overview
This project, titled "From Core to Computation", focuses on deep and machine learning strategies for geological data interpretation in the mythical Arcadia's Avalon Basin.
This project explored the Paleocene deep-marine strata within Quadrants 204 and 205, encompassing significant oil-producing fields and a non-reservoir well (204/24-6).
This project focuses on predicting permeability, classifying the litho-facies, and visualisation in geological formations using machine learning models. By integrating data from core samples and wireline logs across various wells, the goal is to enhance the accuracy of permeability estimations. 

**Key Modules:**
+ Prediction of permeability
+ Classification of lithofacies
+ Visualisation tool for the predicted permeability and facies of the well
+ Notebook demonstrating each component




## Installation Guide

1. Create a clean conda environment with Python 3.11
```bash
conda env create -f environment.yml -n arcadia
```
2. Activate it 
```bash
conda activate arcadia
```
3. Clone the repository by following the below command to your github repository

 git clone https://github.com/ese-msc-2023/ads-arcadia-poropermeables.git
 ```
4. Install package by using: 
```
```
```bash
pip install .
```

<!-- #region -->



## Permeability Predictor
For Permeability Predictor, we have implemented in 'predictor.py' and demonstrated in jupyter notebook 'PredictionNotebook.ipynb'.

### Code Explanation
1. Data Loading

Data is loaded from las and csv files containing permeability and wireline log data. Geolographical information is also added to the datasets.

2. Data Preprocessing

Cleaning the data such as handling missing values, merging wireline log data with core sample data, and replacing special characters.

3. Feature Engineering

Selecting features based on the correlation and geological factors.Specific transformations e.g. logarithmic are applied to some features.

4. Model Training and Evaluation

- Base Models: Utilizes RandomForestRegressor and XGBRegressor as base models.
- Hyperparameter Tuning: Conducts GridSearchCV to find the best hyperparameters for both models.
- Stacked Model: Combines the base models using a StackingRegressor with a LinearRegression model as the final estimator.
- Pipeline: Creates a pipeline that scales the data and applies the stacked model.
- Evaluation: The model is evaluated on both a validation set and a test set, reporting RMSE and R² scores.








<!-- #endregion -->

<!-- #region -->

## Lithofacies classifier

### Overview
This Python-based tool is designed for processing and analyzing core sample images in the petroleum industry. It utilizes machine learning techniques for facies classification, aiding in geological data analysis.

### Features
- **Core Sample Image Processing**: Converts .npy files to images, segments them based on depth, and associates facies labels.
- **Data Aggregation**: Creates a unified dataset from individual processed samples for analysis.
- **Machine Learning Model Training**: Trains a CNN model to classify facies types in core samples.
- **Data Splitting**: Divides data into training, validation, and test sets for model evaluation.
- **Prediction and Evaluation**: Predicts facies types on new core samples and evaluates model performance using confusion matrices.
- **Visualization**: Overlays prediction results on core sample images for a comprehensive visual representation.
|
### Usage
1. **Data Preprocessing**: Use `preprocess` and `process_data` functions to convert and segment core sample images.
2. **Prepare Dataset**: Aggregate processed data using `total_dataframe`.
3. **Split Data**: Divide the dataset into training, validation, and test sets using `split_data`.
4. **Train Model**: Train the CNN model using `train2`.
5. **Predict and Evaluate**: Generate predictions and evaluate the model using `pred` and `prediction_unseen_well`.
6. **Visualize Results**: Visualize the prediction results on core samples with `plot_prediction`.



## Visualisation


### Overview

This tool is designed for geological data analysis, particularly in the context of the petroleum industry. It processes and visualizes well log data, including facies classifications and permeability measurements.

### Features

- **Data Processing**: Reads and processes well log and facies data from CSV files, then maps and filters this data based on depth information.
- **Visualization**: Generates interactive plots showing various aspects of well logs, such as density, resistivity, gamma-ray logs, and more.
- **Facies Classification**: Visualizes different facies types using a color-coded scheme, enhancing the understanding of geological formations.
- **Permeability Analysis**: Includes the capability to visualize permeability data alongside other well log characteristics.
- **Interactive Selection**: Allows users to select different wells from a dropdown menu and dynamically updates the visualizations accordingly.

### Usage

1. **Data Preparation**: Ensure that well log data and facies classification data are available in the specified CSV format.
2. **Running the Tool**: Execute the `handle_data()` function to process the data, followed by `visualize()` to start the visualization interface. The `predict_visualize()` function can be used to display the predicted results.
3. **Interactive Exploration**: Use the dropdown menu to select different wells and explore their geological data through the generated plots.


### Note

This README is a brief overview. For detailed instructions and information, refer to the code documentation and comments.

### Test
<!-- #endregion -->

```bash

```
