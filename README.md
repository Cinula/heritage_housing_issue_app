# Housing Price Prediction

## Project Overview

Lydia Doe has inherited four properties in Ames, Iowa and requires accurate valuation to make informed decisions about selling these assets. Without reliable price estimates based on local market conditions, she risks undervaluing her inherited properties, potentially leaving significant money on the table, or overpricing them, resulting in prolonged market exposure and carrying costs.

The solution will accurately predicts fair market values for Lydia's four inherited properties in Ames, Iowa to provide reliable decision support and incorporate local market specifics of Ames and  Iowa. The application will also predict the house price of any house based in the region based on the Machine Learning (Regression Models) trained.

---

### Objective
To develop a data web application that accurately predicts house sales prices in Ames, Iowa, with a specific focus on valuing Lydia Doe's four inherited properties. The application leverages historical housing data to identify key value drivers in the local market and provides reliable price estimates to maximize potential sales revenue.

---

### Outcomes

#### 1. **Market Analysis**
- Identify and visualize the key attributes that significantly influence house prices in Ames, Iowa.
- Provide data-driven insights into the Ames housing market dynamics.

#### 2. **Predictive Model**
- Develop a robust machine learning model trained on historical Ames housing data.
- Create a user-friendly web interface that allows Lydia to input property attributes and receive price predictions.
- Enable accurate valuation of the four inherited properties.

#### 3. **Decision Support Tool**
- Provide comparative analysis of the four inherited properties against market benchmarks.
- Offer a reusable tool for evaluating future property investments in the Ames and Iowa area.

#### 4. **Documentation**
- Deliver clear documentation explaining model methodology and limitations.
- Include guidance on interpreting results and confidence intervals.
- Provide instructions for ongoing use of the prediction tool.

This solution empowers Lydia to make informed decisions about her inherited properties while giving her a lasting tool to evaluate future property opportunities in the Ames, Iowa market.

---

## Project Structure
```
├── Procfile
├── README.md
├── data
│ ├── house-metadata.txt
│ ├── house_prices_records.csv
│ └── inherited_houses.csv
├── model_building.ipynb
├── model_evaluation_results
│   ├── model_results.csv
│   └── results_inherited_houses_predictions.csv
├── processed_data
│   ├── X.csv
│   └── y.csv
├── requirements.txt
├── repo_screenshots
├── runtime.txt
├── saved_model
│   └── model.pkl
├── setup.sh
├── src
│   └── app.py
```
5 directories, 16 files

---


## Dataset Description

The Ames Housing Dataset is a comprehensive real estate dataset compiled for use in data science education. It describes the sale of individual residential properties in Ames, Iowa with 24 explanatory variables covering virtually every aspect of residential homes, this dataset has become a robust alternative to the Boston Housing dataset for advanced regression techniques in machine learning and statistical modeling.

The dataset used is the **Ames Housing Dataset**, obtained from Kaggle: [Kaggle - Ames Housing](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)

- **Number of rows:** 1,460
- **Features:** 24 columns (both numeric and categorical)
- **Missing data:** Some columns have missing values (e.g., GarageYrBlt, LotFrontage)
- **Target variable:** `SalePrice` (in USD)
- **Independent Features**: 24 explanatory variables
- **Format**: CSV files

![Alt text](/repo_screenshots/dataset_head.JPG?raw=true)


### Dataset Details

The dataset contains a rich variety of features categorized as follows:

- Location Information
- House Features
- Basement Features
- Garage Information
- Outdoor Features
- Sale Information

Data Quality: The dataset is well-documented and relatively clean, though it contains:

- Missing values that require imputation
- A mix of categorical, ordinal, and numerical variables
- Some variables with skewed distributions
- A few outliers that may require special handling


![Alt text](/repo_screenshots/missing_values.JPG?raw=true "Missing Values Plot")


#### Steps Involved for Data Creation for Model Building

Data Analysis, Data cleaning, feature selection, and imputation for missing values were done using Scikit-learn pipelines.

---

## How to Run the Project

### 1. **Local Setup**
1. Clone the repository:
```markdown
git clone https://github.com/Cinula/heritage_housing_issue_app.git
cd heritage_housing_issue_app
```

2. Install dependencies:
```markdown
pip install -r requirements.txt
```

3. Run the Streamlit app:
```markdown
streamlit run src/app.py
```

### 2. **Heroku Deployment**

The application is deployed on Heroku. You can access it using the following link: Heroku App Link

[Housing Price Prediction Web App](https://heritage-housing-app-5e4c533af1d1.herokuapp.com/)

---
## EDA and Model Building Notebook

This is the jupyter notebook with detailed comments and explanation about Exploratory Data Analysis, Feature Engineering, Model Building and Model Evaluation.

[EDA and Model Building Notebook](https://github.com/Cinula/heritage_housing_issue_app/blob/main/model_building.ipynb)

---

## Key Features of the App
- Users can see the home page with detailed information about each tab.
- Users can see the Correlation of top features with SalesPrice of housee.
- Feature Importance: Visualizes the top factors influencing house prices with model evaluaiton results.
- Lydia can see the  predicted price of her four inherited properties.
- Input Form: Users can input property attributes to get price predictions

---

# Bugs and Fixes

##### 1. Outliers in the Data
**Bug:** The dataset contains outliers in several features, such as `GrLivArea`, `LotArea`, and `SalePrice`. These extreme values can distort the model's predictions and reduce accuracy.

**Fix:** Instead of removing outliers (which could lead to loss of valuable information), robust regression algorithms like Random Forest, Extra Trees, and Gradient Boosting were used. These models are less sensitive to outliers and can handle them effectively.

##### 2. Skewed Target Variable
**Bug:** The target variable, `SalePrice`, is positively skewed (skewness = 1.88), which can affect the performance of models that assume normality.

**Fix:** A log transformation was applied to `SalePrice` during model training to reduce skewness and improve model performance. Predictions were then exponentiated back to the original scale for interpretability.

##### 3. Missing Values
**Bug:** Several features in the dataset, such as `LotFrontage`, `GarageYrBlt`, and `MasVnrArea`, contain missing values, which can lead to errors during model training.

**Fix:** Missing values were imputed using appropriate strategies:
- **Numerical Features:** Imputed with the median value.
- **Categorical Features:** Imputed with the most frequent category or "None" where applicable.

##### 4. Feature Encoding
**Bug:** Categorical features like `BsmtExposure`, `GarageFinish`, and `KitchenQual` were not in a format suitable for machine learning models.

**Fix:** These features were encoded using ordinal encoding or one-hot encoding, depending on their nature, to make them compatible with the models.

##### 5. Model Compatibility with Deployment
**Bug:** The `model.pkl` file caused compatibility issues during deployment on Heroku due to differences in the scikit-learn version.

**Fix:** The scikit-learn version used to train the model was explicitly specified in the `requirements.txt` file to ensure compatibility between the local and Heroku environments.

##### 6. Feature Importance Misinterpretation
**Bug:** Users might misinterpret the feature importance visualization as causal relationships rather than correlations.

**Fix:** Clear documentation was added to explain that feature importance reflects the model's reliance on features for predictions, not causality.

##### 7. Deployment Performance
**Bug:** The app's performance on Heroku was initially slow due to the size of the dataset and model.

**Fix:** The dataset was preprocessed and saved as smaller files (`X.csv` and `y.csv`), and the trained model was serialized using pickle for faster loading during deployment.

##### 8. While Using webb App Prediction Tab
**Bug:** The app will throw an matplotlib.backend_agg.RendererAgg error. This is because the web page can not render the simultaneous selected changes of features for prections.

**Fix:** User has to select the values of selected features one by once to see the predicted house price as with every selection of individual feature it makes a prediction.

---

## Limitations

- The model's predictions are based on historical data and may not account for future market changes.
- Confidence intervals are not explicitly provided in the current implementation.

---

## Model Performance Comparison

| Model              | RMSE     | MAE      | MAPE  | Median AE | R²   | Adjusted R² |
|-------------------|----------|----------|------|------------|------|--------------|
| Extra Trees      | 25,242.62 | 16,188.45 | 10.08 | 10,035.50  | 0.909 | 0.903        |
| Gradient Boosting | 27,190.81 | 17,027.44 | 9.96  | 11,005.92  | 0.894 | 0.887        |
| Random Forest    | 28,044.14 | 17,548.29 | 10.43 | 10,063.50  | 0.888 | 0.880        |
| XGBoost         | 28,426.55 | 17,106.00 | 9.98  | 9,535.91   | 0.885 | 0.877        |
| AdaBoost        | 34,061.04 | 23,757.02 | 15.83 | 16,801.89  | 0.834 | 0.823        |


![Alt text](/repo_screenshots/feature_importance.JPG?raw=true "Missing Values Plot")


Lydia Does four houses prediction results:

| House Number              | Predicted Sales Price     |
|-------------------|----------|
| 1      | $121671.82 |
| 2 | $160107.00 |
| 3    | $166636.50 |
| 4         | $190618.00 |



---

## **Recommendation**

The **Extra Trees Regressor** is the most robust and accurate model for predicting house sale prices in Ames, Iowa. Its ability to minimize errors and explain the variance in sale prices makes it the ideal choice for deployment in the web application. This model will provide Lydia Doe with reliable price predictions for her inherited properties and serve as a valuable decision-making tool for future property evaluations.

By leveraging the Extra Trees Regressor, Lydia can confidently assess the value of her properties and make informed decisions to maximize potential sales revenue.

---

## Future Enhancements

- Add confidence intervals for predictions.
- Integrate additional data sources for improved accuracy.
- Expand the app to support other housing markets.