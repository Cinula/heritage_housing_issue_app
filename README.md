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

## Business Requirements

Below are the detailed business requirements categorized into Data Visualization and Machine Learning tasks.

### 1. Data Visualization Requirements

| ID | Description | User Story |
|----|-------------|------------|
| DV_1 | Correlation Analysis | visualize the correlation between different house features and price, so that I can understand which factors most influence property values. |
| DV_2 | Price Trends Over Time | As an investor, I want to visualize how house prices have changed over time, so that I can make informed decisions about market trends. |
| DV_3 | Feature Importance Visualization | As a property developer, I want to see which house features add the most value, so that I can focus on these aspects in new developments. |
| DV_4 | Price Range Distribution | As a real estate agent, I want to visualize the distribution of houses across price ranges, so that I can better advise clients on competitive pricing. |

### 2. Machine Learning Requirements

| ID | Description | User Story |
|----|-------------|------------|
| ML_1 | Price Prediction Model | As a homebuyer, I want an accurate prediction of a house's value based on its features, so that I can determine if it's fairly priced. |
| ML_2 | Feature Importance Analysis | As a property seller, I want to know which features most affect my home's value, so that I can focus on improving these aspects before selling. |
| ML_4 | Model Building | As a data scientist, I want to train and build different Machine Learning Model to predict the Sale price of a house. |
| ML_3 | Model Comparison and Evaluation | As a data scientist, I want to compare different regression models, so that I can identify the most accurate approach for house price prediction. |

### 3. Implementation Plan

Each requirement will be addressed through the following approaches:

**Data Visualization:**
- Interactive dashboards using Matplotlib, Seaborn, and Plotly
- Time series analysis with trend visualization
- Correlation heatmaps and feature importance charts

**Machine Learning:**
- Implementation of multiple regression models (Linear Regression, Random Forest, XGBoost etc.)
- Hyperparameter tuning for optimization
- Feature engineering based on correlation analysis
- Model evaluation using RMSE, MAE, and R² metrics

**Streamlit web app**
- Interactive streamlit web app to show results and charts.
- Tool to predict sale price for a new house based on inputs by user.
- Depoyment of streamlit web app on Heroku.

### 4. Success Criteria

1. Model achieves prediction accuracy with low RMSE and high R².
2. Clear visualization of at least 5 key factors influencing house prices
3. Interactive features allowing users to input house characteristics and receive price estimates
4. Comprehensive documentation of model limitations and reliability across different market segments

---



## Dataset Description

The Ames Housing Dataset is a comprehensive real estate dataset compiled for use in data science education. It describes the sale of individual residential properties in Ames, Iowa with 24 explanatory variables covering virtually every aspect of residential homes, this dataset has become a robust alternative to the Boston Housing dataset for advanced regression techniques in machine learning and statistical modeling.

The dataset used is the **Ames Housing Dataset**, obtained from Kaggle: [Kaggle - Ames Housing](https://www.kaggle.com/datasets/codeinstitute/housing-prices-data)

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

#### Steps Involved for Data Creation for Model Building

Data Analysis, Data cleaning, feature selection, and imputation for missing values were done using Scikit-learn pipelines.

---

## Data Analysis

1. **Missing Values**

![Alt text](/repo_screenshots/missing_values.JPG?raw=true "Missing Values Plot")

We are checking the missing value in our dataset. As we can see, there are some features that has missing values in them. In order to have a good model to predict results, it is important to impute missing values. We have set the threshold of mssing values to 80%. Any features having more than 80% missing values will be removed from our analysis and model training.


2. **Sales Price vs Years Features**

![Alt text](/repo_screenshots/sale_price_vs_Years.JPG?raw=true)

In the above charts we are seeing a house Sale price trend over the periods. In all the three features, the house price is in increasing trend over the years of Garage Built, the year of house built and the year the house is remodelled.

3. **Median House Price**

![Alt text](/repo_screenshots/median_house_price.JPG?raw=true)

We are seeing the median house Sale price trend over the periods. We can see an increasing trends for the overall period with little fluctuations between 1940 to 1980. There is a suddden increase in average house prices between 1980 as the bank rates of interests sky rocketed during that time hence affecting the mortgage rates for the borrowers.

4. **Correlation Analysis**

![Alt text](/repo_screenshots/correlation.JPG?raw=true)

Here we see that the OverallQual feature is nearly 80% correlated with the target variable. Overallqual feature refers to the overall material and quality of the materials of the completed house. Well, this make sense as well. People usually consider these parameters for their dream house. In addition, GrLivArea is nearly 70% correlated with the target variable. GrLivArea refers to the living area (in sq ft.) above ground. The following variables show people also care about if the house has a garage, the area of that garage, the size of the basement area, etc.

5. **Distribution of Sales Price (Target Variable)**

![Alt text](/repo_screenshots/distribution_sale_price.JPG?raw=true)

We can see that the target is continuous, and the distribution is skewed towards the right. The skewness of the SalePrice is 1.88. This indicates that the distribution of house prices is positively skewed (or skewed to the right). The distribution has a longer tail on the right side. Most of the house prices are concentrated on the lower end of the scale, with fewer houses having very high prices. The right tail suggests the presence of outliers or extreme values (houses with very high sale prices). The target variable does not follow a normal distribution, which can affect the performance of certain machine learning models that assume normality (e.g., linear regression).

6. **Pair Plots**

![Alt text](/repo_screenshots/pair_plots.JPG?raw=true)

The top 5 highly correlated features with Sales price has been plotted above. These are OverallQual, GrLivArea, GarageArea, TotalBsmtSF, 1stFlrSF. We can see that the features are linearly correlated with the target variable in the above scatter pair plots.

7. **Box Plots**

![Alt text](/repo_screenshots/box_1.JPG?raw=true) ![Alt text](/repo_screenshots/box_2.JPG?raw=true)

The Box plot shows us the outliers (extreme values) in the features. We have seen that in our catrgorical features, there are quite a few outliers. However, we are not removing or treating them. Insteat we will be using Robust machine learning regressors that are not sensitive to outliers and can handle them easily during the training and predictions.

8. **Distribution Plots for Independent Variables**

![Alt text](/repo_screenshots/distribution_1.JPG?raw=true)

We have created similar distribution plots for the other variables that could be found in this [EDA and Model Building Notebook](https://github.com/Cinula/heritage_housing_issue_app/blob/main/model_building.ipynb) notebook here. - The above shows distribution spread of our numerical features. Some of the features are highly positively and negatively skewed. This can be handled in data preprocessing by doing certain transaformations on these columns. Alternatively, ensemble regressors doesn't need to normalise and transform the data. Hence, we will use them in our model training.

9. **Scatter Plots**

![Alt text](/repo_screenshots/scatter.JPG?raw=true)

We have created similar scatter plots for the other variables that could be found in this [EDA and Model Building Notebook](https://github.com/Cinula/heritage_housing_issue_app/blob/main/model_building.ipynb) notebook here. Scatter plots above shows us the relationship of our features with the target variable SalePrice. Some of the features that are linearly correlated to Saleprice could be seen showing us linear trend of data. While the ones that doesn't have strong correlation doesn't show any pattern/trend and the data points are randomly scattered around the axis. 

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

## Hypothesis Testing

There could be many hypothesis created around this project. I have decided to consider LotArea and Sales price as an example to support by business objective with the one of the below mentioned hypothesis.

- Null Hypothesis (H0): LotArea do not significantly affect the sale price.
- Alternate Hypothesis (H1): LotArea significantly affect the sale price.

![Alt text](/repo_screenshots/hypothesis.JPG?raw=true)

#### Explanation:

**p-value=1.1231391549185238e-24**

1. P-value: The p-value for LotArea is extremely small (p < 0.001), indicating strong statistical significance.

2. Coefficient: For each additional square foot of lot area, the sale price increases by approximately $2.10.

3. R-squared: The model explains about 7% (R² = 0.070) of the variation in sale prices, suggesting that while lot area is a significant predictor, other factors not included in this model also strongly influence home prices.

4. F-statistic: The high F-statistic (109.1) with a very low probability (1.12e-24) further confirms the statistical significance of the relationship.

While the relationship is statistically significant, the relatively low R-squared value suggests that lot area alone is not sufficient to comprehensively predict home sale prices, and a more complete model would likely include additional variables. Since p-value is extremely small, we reject the null hypothesis and conclude that independent variable LotArea has statistically significant impact on the dependent variable SalePrice.

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


**Feature Importance (Extra Trees)**

![Alt text](/repo_screenshots/feature_importance.JPG?raw=true)


**Actual vs Predicted SalesPrice (Extra Trees)**

![Alt text](/repo_screenshots/extra_tree_actual_predicted.JPG?raw=true)




**Lydia Does four houses prediction results:**

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