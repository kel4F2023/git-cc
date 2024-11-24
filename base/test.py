import joblib

# Load the trained pipeline
pipeline = joblib.load('../src/git_cc/model/base_classifier.joblib')

# Predict commit type
message = "added setup.py for building cli tool"
commit_type = pipeline.predict([message])[0]
print(f"Predicted commit type: {commit_type}")
