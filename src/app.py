import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Import necessary libraries
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load the trained model
model = pickle.load(open(os.path.join(os.path.dirname(__file__), '../saved_model/model.pkl'), 'rb'))

# Load your dataset (used for correlation heatmap)
# Replace 'your_dataset.csv' with the actual dataset file
# data = pd.read_csv('../data/house_prices_records.csv')


data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/house_prices_records.csv'))

# results_df = pd.read_csv('../model_evaluation_results/model_results.csv')
results_df =  pd.read_csv(os.path.join(os.path.dirname(__file__), '../model_evaluation_results/model_results.csv'))

# X = pd.read_csv('../processed_data/X.csv')
# y = pd.read_csv('../processed_data/y.csv')

X = pd.read_csv(os.path.join(os.path.dirname(__file__), '../processed_data/X.csv'))
y = pd.read_csv(os.path.join(os.path.dirname(__file__), '../processed_data/y.csv'))





# inherited_house_df = pd.read_csv('../results_inherited_houses_predictions.csv')
inherited_house_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../model_evaluation_results/results_inherited_houses_predictions.csv'))

# Title and description
st.title("Heritage Housing Issue")
st.write("This intelligent system predicts the sale price of a house and provides insights into feature correlations.")

# Create tabs
tab_0, tab1, tab2, tab3, tab4 = st.tabs(["Home Page", "Correlation Heatmap and Distribution of SalePrice", "Model Evaluation Results", "Inherited Houses Price Prediction", "New Ames/IowaHouse Price Prediction"])


# Tab 0: Home Page
with tab_0:
    st.header("Home Page")
    st.write("Welcome to the House Sale Price Prediction App!")
    st.write("This app predicts the sale price of a house based on various features such as the number of bedrooms, living area, garage area, and more.")
    st.write("Navigate to the other tabs to view the correlation heatmap, model evaluation results, inherited houses price predictions, and predict the sale price of a new house.")

    st.write("\n")
    st.write("To get started, navigate to the other tabs to explore the features and predictions!")

    st.write("\n")
    st.write("1. The correlation heatmap and model evaluation results tab provides a correlation heatmap showing the correlation between numerical features and the target variable (SalePrice). The tab also includes the top 10 features that are most positively correlated with SalePrice.")

    st.write("\n")
    st.write("2. The model evaluation results tab provides insights into the performance of different regression models, including Random Forest, XGBoost, AdaBoost, Extra Trees, and Gradient Boosting. The tab includes the root mean squared error (RMSE), mean absolute error (MAE), mean absolute percentage error (MAPE), median absolute error (Median AE), R2, and adjusted R2 values for each model.")

    st.write("\n")
    st.write("3. The inherited houses price prediction tab displays the predicted sale price of inherited houses based on the trained model.")

    st.write("\n")
    st.write("4. The new Ames/IowaHouse price prediction tab allows users to input house details such as the number of bedrooms, living area, garage area, etc., to predict the sale price of a new house.")

    


# Tab 1: Correlation Heatmap and Distribution of SalePrice
with tab1:
    st.header("Correlation Heatmap and Distribution of SalePrice")
    st.write("This heatmap shows the correlation between numerical features and the target variable (SalePrice).")


    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr, annot=True, fmt=".2")
    st.pyplot(plt)

    st.write("\n")
    st.write("Top 10 features that are most positively correlated with SalePrice:")
    st.write(corr['SalePrice'].sort_values(ascending=False)[1:11]) #top 10 correlations

    st.write("This chart shows the Distribution of target variable (SalePrice).")


    st.write("We can see that the target is continuous, and the distribution is skewed towards the right. The skewness of the SalePrice is 1.88.")


    # Visualize the distribution of the target variable (SalePrice)
    plt.figure(figsize=(10, 6))
    sns.histplot(data['SalePrice'], kde=True, bins=30, color='blue')
    plt.title('Distribution of SalePrice')
    plt.xlabel('SalePrice')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    st.write("This indicates that the distribution of house prices is positively skewed (or skewed to the right.")

    st.write("The distribution has a longer tail on the right side. Most of the house prices are concentrated on the lower end of the scale, with fewer houses having very high prices.")

    st.write("Outliers: The right tail suggests the presence of outliers or extreme values (houses with very high sale prices).")

    st.write("Non-Normal Distribution: The target variable does not follow a normal distribution, which can affect the performance of certain machine learning models that assume normality (e.g., linear regression.")

# Tab 2: Model Evaluation Results
with tab2:    
    st.header("Model Evaluation Results")
    st.write("The table below shows the evaluation results of the trained model.")
    st.write(results_df)

    best_model = results_df.loc[results_df['RMSE'].idxmin()]
    st.write(f"\nBest Model Based on RMSE and Adjusted R2: {best_model['Model']}")
    st.write(f"RMSE: {best_model['RMSE']:.2f}, MAE: {best_model['MAE']:.2f}, MAPE: {best_model['MAPE']:.2f}%, Median AE: {best_model['Median AE']:.2f}")
    st.write(f"R2: {best_model['R2']:.2f}, Adjusted R2: {best_model['Adjusted R2']:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    model = models[best_model['Model']]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_test = y_test.squeeze()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
    plt.title(f'Actual vs. Predicted SalePrice ({best_model["Model"]})')
    plt.xlabel('Actual SalePrice')
    plt.ylabel('Predicted SalePrice')
    plt.grid(True)
    # plt.show()
    st.pyplot(plt)


    # Extract feature importances
    feature_importances = model.feature_importances_

    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })

    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot the top 10 most important features
    plt.figure(figsize=(15, 10))
    plt.barh(feature_importance_df['Feature'][:][::-1], feature_importance_df['Importance'][:][::-1], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Top 10 Most Important Features (Extra Trees)')
    # plt.show()
    st.pyplot(plt)

    

    st.write("\n")
    st.write(f"""The {best_model['Model']} Regressor demonstrates strong predictive performance with an RMSE of {round(results_df.loc[results_df['Model'] == best_model['Model']]['RMSE'].values[0], 2)} 
             and an MAE of {round(results_df.loc[results_df['Model'] == best_model['Model']]['MAE'].values[0], 2)}, 
             indicating that the model's predictions are reasonably close to the actual sale prices. The MAPE of {round(results_df.loc[results_df['Model'] == best_model['Model']]['MAPE'].values[0], 2)}% shows that the model's predictions are, on average, 
             within {round(results_df.loc[results_df['Model'] == best_model['Model']]['MAPE'].values[0], 2)}% of the actual values. The Median AE of {round(results_df.loc[results_df['Model'] == best_model['Model']]['Median AE'].values[0], 2)} suggests that the model's median prediction error is relatively low.
             The high R2 value of {round(results_df.loc[results_df['Model'] == best_model['Model']]['R2'].values[0], 2)} and 
             Adjusted R2 of {round(results_df.loc[results_df['Model'] == best_model['Model']]['Adjusted R2'].values[0], 2)} indicate that the model explains a significant portion of the variance in the sale prices, suggesting a strong fit. 
             Overall, the {best_model['Model']} Regressor is a robust model for predicting house sale prices with high accuracy and reliability.""")



# Tab 3: Inherited Houses Price Prediction

with tab3:
    st.header("Inherited Houses Price Prediction")
    st.write("The table below shows the predicted sale price of inherited houses.")
    st.write(inherited_house_df)

    price_predicted = list(inherited_house_df.Predicted_SalePrice)

    for ind, price in enumerate(price_predicted, start=1):
        st.write(f'House No {ind} and Predicted Price is ${price}')



# Tab 4: New Ames/IowaHouse Price Prediction
with tab4:
    st.header("Predict House Sale Price")
    st.write("Provide the house details in the sidebar to predict the sale price.")

    # Input form for user to provide house details
    st.sidebar.header("Input Features (Select last Tab)")
    def user_input_features():
        firstFlrSF = st.sidebar.number_input("First Floor square feet", 0, 5000, 1500)
        secondFlrSF = st.sidebar.number_input("Second Floor square feet", 0, 5000, 500)
        BedroomAbvGr = st.sidebar.slider("Bedrooms Above Grade", 0, 10, 3)
        BsmtExposure = st.sidebar.selectbox("Basement Exposure", ['None', 'No', 'Mn', 'Av', 'Gd'])
        BsmtFinSF1 = st.sidebar.number_input("Type 1 finished square feet", 0, 5000, 500)
        BsmtFinType1 = st.sidebar.selectbox("Basement Finish Type 1", ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'])
        BsmtUnfSF = st.sidebar.number_input("Unfinished square feet of basement area", 0, 5000, 500)
        GarageArea = st.sidebar.number_input("Garage Area (square feet)", 0, 2000, 500)
        GarageFinish = st.sidebar.selectbox("Garage Finish", ['None', 'Unf', 'RFn', 'Fin'])
        GarageYrBlt = st.sidebar.number_input("Garage Year Built", 1900, 2025, 2005)
        GrLivArea = st.sidebar.number_input("Above grade (ground) living area square feet", 0, 5000, 1500)
        KitchenQual = st.sidebar.selectbox("Kitchen Quality", ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
        LotArea = st.sidebar.number_input("Lot size in square feet", 0, 100000, 10000)
        LotFrontage = st.sidebar.number_input("Lot Frontage (Linear feet of street connected to property)", 0, 500, 80)
        MasVnrArea = st.sidebar.number_input("Masonry veneer area in square feet", 0, 1000, 100)
        OpenPorchSF = st.sidebar.number_input("Open porch area in square feet", 0, 500, 50)
        OverallCond = st.sidebar.slider("Overall Condition", 1, 10, 5)
        OverallQual = st.sidebar.slider("Overall Quality", 1, 10, 5)
        TotalBsmtSF = st.sidebar.number_input("Total square feet of basement area", 0, 5000, 1000)
        YearBuilt = st.sidebar.number_input("Year Built", 1900, 2025, 2000)
        YearRemodAdd = st.sidebar.number_input("Year Remodeled", 1900, 2025, 2000)

        # Create a dictionary of inputs
        data = {
            'FirstFlrSF': firstFlrSF,
            'SecondFlrSF': secondFlrSF,
            'BedroomAbvGr': BedroomAbvGr,
            'BsmtExposure': BsmtExposure,
            'BsmtFinSF1': BsmtFinSF1,
            'BsmtFinType1': BsmtFinType1,
            'BsmtUnfSF': BsmtUnfSF,
            'GarageArea': GarageArea,
            'GarageFinish': GarageFinish,
            'GarageYrBlt': GarageYrBlt,
            'GrLivArea': GrLivArea,
            'KitchenQual': KitchenQual,
            'LotArea': LotArea,
            'LotFrontage': LotFrontage,
            'MasVnrArea': MasVnrArea,
            'OpenPorchSF': OpenPorchSF,
            'OverallCond': OverallCond,
            'OverallQual': OverallQual,
            'TotalBsmtSF': TotalBsmtSF,
            'YearBuilt': YearBuilt,
            'YearRemodAdd': YearRemodAdd
        }

        temp_df = pd.DataFrame([data])
        temp_df = temp_df.rename(columns={'FirstFlrSF': '1stFlrSF', 'SecondFlrSF': '2ndFlrSF'})
        return temp_df

    # Get user input
    input_df = user_input_features()

    # Preprocess the input data (apply the same preprocessing as in your notebook)
    bsmt_exposure_mapping = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}
    bsmt_fin_type1_mapping = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}
    kitchen_qual_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}

    input_df['BsmtExposure'] = input_df['BsmtExposure'].map(bsmt_exposure_mapping)
    input_df['BsmtFinType1'] = input_df['BsmtFinType1'].map(bsmt_fin_type1_mapping)
    input_df['KitchenQual'] = input_df['KitchenQual'].map(kitchen_qual_mapping)

    # One-hot encode GarageFinish
    garage_finish_dummies = pd.get_dummies(input_df['GarageFinish'], prefix='GarageFinish')
    input_df = pd.concat([input_df, garage_finish_dummies], axis=1)
    input_df.drop('GarageFinish', axis=1, inplace=True)


# None, Rfn, Unf
    # print(input_df)
    # Ensure all one-hot encoded columns match the training data
    for col in ['GarageFinish_None', 'GarageFinish_RFn', 'GarageFinish_Unf']:
        if col not in input_df.columns:
            input_df[col] = False  # Add missing columns with default value 0


    input_columns = list(input_df.columns)
    # print(input_columns)
    final_column_list = ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinType1', 'BsmtUnfSF', 'GarageArea', 'GarageYrBlt', 'GrLivArea', 'KitchenQual', 'LotArea', 'LotFrontage', 'MasVnrArea', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd', 'GarageFinish_None', 'GarageFinish_RFn', 'GarageFinish_Unf']

    input_df = input_df[final_column_list]
    # print('before', input_df)
    # Predict the sale price
    prediction = model.predict(input_df)

    # Display the prediction
    st.subheader("Predicted Sale Price")
    st.write(f"${prediction[0]:,.2f}")