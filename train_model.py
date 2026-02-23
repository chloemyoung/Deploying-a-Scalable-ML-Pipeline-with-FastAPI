import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# -------------------------
# Load the census data
# -------------------------

# Project root is the current directory
project_path = os.getcwd()

data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)

data = pd.read_csv(data_path)

# -------------------------
# Train / test split
# -------------------------

train, test = train_test_split(data, test_size=0.20, random_state=42)

# -------------------------
# Categorical features
# -------------------------

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# -------------------------
# Process training data
# -------------------------

X_train, y_train, encoder, lb = process_data(
    X=train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# -------------------------
# Process test data
# -------------------------

X_test, y_test, _, _ = process_data(
    X=test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# -------------------------
# Train the model
# -------------------------

model = train_model(X_train, y_train)

# -------------------------
# Save model and encoder
# -------------------------

model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.pkl")
save_model(model, model_path)

encoder_path = os.path.join(model_dir, "encoder.pkl")
save_model(encoder, encoder_path)

# -------------------------
# Load the model
# -------------------------

model = load_model(model_path)

# -------------------------
# Run inference on test data
# -------------------------

preds = inference(model, X_test)

# -------------------------
# Print overall metrics
# -------------------------

p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# -------------------------
# Slice performance
# -------------------------

# Clear old slice output
with open("slice_output.txt", "w") as f:
    f.write("")

for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]

        p, r, fb = performance_on_categorical_slice(
            data=test,
            column_name=col,
            slice_value=slicevalue,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model,
        )

        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(
                f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n",
                file=f,
            )
