# **Week 2 Report : Predictive Model Building & Evaluation**

---

This document details the complete analysis performed during Week 2 of the project, corresponding to the "Model Building, Training, Evaluation & Serialization" milestone.

All code and analysis described here can be found in the Model\_Building.ipynb Jupyter Notebook. 

## **1\. Objective** 

* Following the data cleaning and feature engineering in Week 1, the objective for Week 2 was to build, train, and evaluate machine learning models to accurately forecast the three key target variables of the microgrid :   
  * load\_demand   
  * solar\_output   
  * wind\_output 

## **2\. Methodology & Code Analysis** 

* The notebook follows a structured approach to model development, ensuring that the data is handled correctly for a time-series problem and that the models are robustly evaluated. 

### **Step 2.1 : Data Loading & Preparation** 

* **Load Data :** The analysis begins by loading the processed\_dataset.csv file, which was the final output from Week 1\. This dataset includes all the engineered temporal, lag, and rolling-average features.   
* Define Features (X) and Targets (y) :   
  * Three separate feature sets (X) and target variables (y) were defined, as each target is predicted by a different set of relevant features.   
  * **Load Model Features :** solar\_irradiance, temperature, hour, day\_of\_year, load\_demand\_lag1, load\_roll\_3h   
  * **Solar Model Features :** solar\_irradiance, temperature, humidity, hour, day\_of\_year, solar\_irradiance\_lag1, solar\_irradiance\_roll\_3h   
  * **Wind Model Features :** wind\_speed, temperature, pressure, wind\_speed\_lag1, wind\_speed\_roll\_3h 

### **Step 2.2 : Feature Scaling** 

* A StandardScaler was employed to normalize the features.   
* The scaler was fit only on the training data (X\_train) and then used to transform both X\_train and X\_test. This prevents "data leakage," where information from the test set could improperly influence the model.   
* The fitted scaler was saved as scaler.joblib for later use in production. 

### **Step 2.3 : Train-Test Split & Visualization** 

#### **A. Time-Series Split :** 

* The data was split into training (80%) and testing (20%) sets.   
* Crucially, shuffle=False was used during the split. This is essential for time-series data, as it ensures that the model is trained on past data (the first 80% of the dataset) and tested on the most recent, "future" data (the final 20%), simulating a real-world forecasting scenario. 

#### **B. Split Visualization :** 

* The graph below clearly shows this split, with the blue line representing the training data used to build the model and the orange line representing the unseen test data used for evaluation. 

![](https://github.com/DhammadeepRamteke/hybrid-microgrid-controller/blob/dafa4ec92f8c0c12bd22cdd4b0a02412254c3caf/Week%202/train_test_split.png)
Train/Test Split Visualization

### **Step 2.4 : Model Selection & Training** 

* **Algorithm :** RandomForestRegressor was chosen as the modeling algorithm. This is a powerful, tree-based ensemble model that is robust to outliers and effective at capturing complex, non-linear relationships.   
* **Model Training :** Three separate Random Forest models were trained—one for each target variable (load\_demand, solar\_output, wind\_output).   
* **Parameters :** All models were trained with n\_estimators=100 (100 decision trees) and min\_samples\_split=10 to help prevent overfitting. 

### **Step 2.5 : Model Performance Evaluation** 

* The performance of each model was evaluated on the unseen test data using three standard regression metrics. The results were highly successful :   
  * **Mean Absolute Error (MAE) :** The average absolute difference between the predicted and actual values.   
  * **Mean Squared Error (MSE) :** The average of the squared errors, which penalizes larger errors more heavily.   
  * **R-squared (R²) :** A measure of how much of the variance in the target variable is explained by the model (1.0 is a perfect score). 

| Model | Target Variable | MAE | MSE | R-squared (R²) |
| ----- | ----- | ----- | ----- | ----- |
| Load Model | load\_demand | 13.91 | 373.80 | 0.98 |
| Solar Model | solar\_output | 4.88 | 51.52 | 0.97 |
| Wind Model | wind\_output | 2.59 | 10.92 | 0.99 |

#### **Conclusion :** 

* The R² scores, all at 0.97 or higher, indicate that the models are extremely effective and can explain the vast majority of the variance in their respective targets. The low MAE and MSE values further confirm their high predictive accuracy. 

### **Step 2.6 : Feature Importance Analysis** 

* A key advantage of Random Forest is its ability to report on feature importance. The analysis revealed the most influential drivers for each prediction :   
  * **For Load Demand :** The most critical feature was load\_demand\_lag1 (the load from the previous hour), followed closely by load\_roll\_3h (the 3-hour rolling average). This confirms that recent load is the strongest predictor of future load.   
  * **For Solar Output :** As expected, solar\_irradiance was the dominant feature. The engineered features solar\_irradiance\_roll\_3h and solar\_irradiance\_lag1 were also highly significant.   
  * **For Wind Output :** The most important feature by a large margin was wind\_speed\_lag1 (the wind speed from the previous hour), followed by the 3-hour rolling average wind\_speed\_roll\_3h. 

### **Step 2.7: Model Serialization** 

* As the final step, the three trained models were saved to disk using joblib :   
  * model\_load.joblib   
  * model\_solar.joblib   
  * model\_wind.joblib   
* These files, along with the saved scaler.joblib, allow the models to be easily reloaded for future predictions without needing to be retrained. 
