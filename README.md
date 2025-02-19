# Predicting-Hazardous-NEOs-Nearest-Earth-Objects-
******📌 Project Overview:******

This project aims to predict whether a Near-Earth Object (NEO) is hazardous using machine learning techniques. The dataset is processed, balanced using SMOTE, and trained using models like Gradient Boosting and XGBoost. Model performance is evaluated with metrics like accuracy, confusion matrix, and classification reports.

**📂 Dataset:**

The dataset used in this project contains information about NEOs collected between 1910 and 2024. The key features in the dataset include:

Absolute Magnitude
Diameter (in km)
Velocity (km/s)
Miss Distance (Earth Distance Units - AU, LD, KM)
Orbiting Body
Potentially Hazardous (Target variable: 0 = Not Hazardous, 1 = Hazardous)

**📊 Data Preprocessing & Cleaning:**

  1. Load the dataset and inspect missing values.
  2. Removed unnecessary columns (neo_id, name, orbiting_body).
  3. checking for duplicate enteries and found no duplicates.
  4. Encoded categorical variables "is_hazardous" using LabelEncoder.
  5. Standardize the features using StandardScaler.

**⚙️ Handling Class Imbalance with SMOTE:**

Since the dataset is imbalanced, Synthetic Minority Over-sampling Technique (SMOTE) was applied to generate synthetic samples of the minority class before splitting the dataset into training and testing sets.

**🔍 Exploratory Data Analysis (EDA):**

  1. Class Distribution: Visualized using a count plot to check data imbalance.
  2. Correlation Matrix: Heatmap to identify relationships between variables.
  3. Pair Plot: To observe feature distributions and patterns.

**🚀 Machine Learning Models Used:**

1️⃣ Random Forest Classifier
  Applied after SMOTE oversampling.
  Evaluated with Accuracy, Confusion Matrix, and Classification Report.
  Giving accuracy: 0.9514799217035411

2️⃣ Gradient Boosting Classifier
  Applied with hyperparameter tuning (n_estimators=100, learning_rate=0.1, max_depth=3)
  Improved performance on imbalanced data.
  Giving accuracy:  0.7534558464791022  which is the worthest acuuracy

3️⃣ Stacking Classifier (Ensemble Learning)
  Combines Random Forest and XGBoost with Logistic Regression as the final estimator.
  Giving accuracy: 0.9506156101446451

**📈 Model Evaluation:**

  Accuracy Score to measure the overall performance.
  Confusion Matrix to visualize True Positives, True Negatives, False Positives, and False Negatives.
  Classification Report to analyze Precision, Recall, and F1-score for each class.

**🔥 Example of Model Output:**


![image](https://github.com/user-attachments/assets/373b2864-328b-4f3d-a647-b6ccd206dcd8)

**🛠️ Technologies Used**

Python (pandas, numpy, seaborn, matplotlib)
Scikit-Learn (Random Forest, Gradient Boosting, Stacking Classifier)
Imbalanced-learn (SMOTE)
