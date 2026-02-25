# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Model type:** Random Forest Classifier  
- **Framework:** scikit-learn  
- **Model version:** v1.0  
- **Input:** Tabular census data with categorical and numerical features  
- **Output:** Binary classification predicting income level (`<=50K` or `>50K`)  

The model uses one-hot encoding for categorical variables and a label binarizer for the target variable. It was trained using supervised learning on labeled census data.


## Intended Use
This model is intended **for educational purposes only**.

It is designed to:
- Demonstrate an end-to-end machine learning pipeline
- Practice MLOps concepts such as testing, model serialization, and deployment
- Serve predictions via a REST API using FastAPI

This model is **not intended** for:
- Real-world income prediction
- Automated decision-making systems
- Use in hiring, lending, or financial decisions

## Training Data
The model was trained on the **Census Income Dataset**, which contains demographic and employment-related information such as:

- Age  
- Workclass  
- Education  
- Marital status  
- Occupation  
- Relationship  
- Race  
- Sex  
- Capital gain and loss  
- Hours worked per week  
- Native country  

The target variable is **salary**, indicating whether an individual earns more than $50,000 per year.

The dataset was split into training and testing sets using a standard train-test split.

## Evaluation Data
The evaluation data consists of a held-out test set from the same Census dataset.

In addition to overall evaluation metrics, the model was evaluated on **slices of the data** based on categorical features (e.g., sex, race, occupation) to assess performance consistency across subgroups.

## Metrics
The model was evaluated using the following metrics:

- **Precision**
- **Recall**
- **F1 Score**

Example performance on the test dataset:
- Precision: ~0.74  
- Recall: ~0.64  
- F1 Score: ~0.69  

Slice-based performance metrics were also calculated to identify potential disparities across different categorical groups.

## Ethical Considerations
The dataset includes sensitive demographic attributes such as **race**, **sex**, and **marital status**.

Potential ethical concerns include:
- Reinforcement of historical or societal biases
- Unequal performance across demographic subgroups
- Inappropriate use in high-stakes decision-making contexts

This model should not be deployed without further bias and fairness analysis.

## Caveats and Recommendations
- The model is trained on historical census data and may reflect existing biases.
- Performance may not generalize well to populations outside the training distribution.
- The model does not account for temporal changes in income patterns.
- Additional fairness audits and bias mitigation techniques are recommended before any real-world deployment.

This model is best used as a **learning tool** to demonstrate machine learning deployment workflows.