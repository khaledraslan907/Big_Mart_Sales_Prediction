# Big_Mart_Sales_Prediction
# XGBRegressor model conducted hyperparameter tuning using GridSearchCV
# Project Overview
# 1- Importing Libraries and Data:
We began by importing the necessary Python libraries for data manipulation, visualization, and modeling, including pandas, numpy, matplotlib, seaborn, and xgboost. We then loaded the dataset from a CSV file into a Pandas DataFrame.
# 2- Data Description and Initial Analysis:
We explored the dataset to understand its structure and content. This included checking the first few rows, getting summary statistics, and identifying any missing values. The dataset contained various features related to products and outlets, with Item_Outlet_Sales being the target variable.
# 3- Handling Missing Values:
We handled missing values to prepare the data for modeling. Specifically:
Filled missing Item_Weight values with the mean weight.
Filled missing Outlet_Size values with the mode (most frequent value).
# 4- Data Analysis and Plotting:
We conducted exploratory data analysis (EDA) to uncover insights and visualize relationships within the data:
Visualized the distribution of the target variable Item_Outlet_Sales.
Analyzed the distribution of categorical variables like Item_Type, Outlet_Type, and Outlet_Location_Type.
Investigated correlations between numerical features and the target variable.
# 5- Data Preprocessing:
To prepare the data for machine learning models, we performed Encoded categorical variables using LabelEncoder to convert them into numerical format.
# 6- Train-Test Split:
We split the dataset into training and testing sets to evaluate the model’s performance on unseen data. We used an 80-20 split, with 80% of the data used for training and 20% for testing.
# 7- Model Training and Evaluation:
We trained an XGBRegressor model, an implementation of gradient boosting, which is suitable for regression tasks:
Performed initial training with default parameters and evaluated the model using the R-squared metric.
Conducted hyperparameter tuning using GridSearchCV to find the best combination of hyperparameters, which improved the model’s performance.
Evaluated the tuned model’s performance on both the training and testing datasets.
# 8- Results:
The final model’s R-squared values indicated the proportion of variance in the target variable that was explained by the features:
Training R-squared: 0.63
Testing R-squared: 0.59
These results demonstrated an improvement over the initial model, indicating that hyperparameter tuning effectively reduced overfitting and improved generalization.
