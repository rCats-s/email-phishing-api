from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import opendatasets as od
import featureextraction as fe

app = FastAPI(title="Phishing Email Detection API")

# Load scaler
scaler = joblib.load('scaler.pkl')

data_df = pd.read_csv('./email_phishing_data.csv')
cleaned_data_df = data_df.dropna().loc[:, data_df.columns != 'label']
input_size = cleaned_data_df.shape[1]

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Load model
model = MyModel(input_size).to(device)
model.load_state_dict(torch.load('phishing_model.pth', map_location=device))

# Disable batchnorm for single-sample inference
model.bn1 = nn.Identity()
model.bn2 = nn.Identity()
model.eval()

class EmailInput(BaseModel):
    text: str = Field(...,
        description="The email text to classify. "
                    "Please mark hyperlinks using '--link--' so they can be correctly identified."
    )

@app.post("/predict")
def predict_phishing(input_data: EmailInput):
    try:
        features = fe.extract_features(input_data.text)
        features = features.astype(np.float32)
        features = np.log1p(features)
        features = scaler.transform(features.reshape(1, -1))
        features = torch.tensor(features, dtype=torch.float32).to(device)

        if features.shape[1] != input_size:
            raise HTTPException(status_code=400, detail="Feature size mismatch with model input.")

        with torch.no_grad():
            prediction = torch.sigmoid(model(features))
            pred_label = 1 if prediction.item() >= 0.5 else 0

        return {
            "phishing_probability": float(prediction.item()),
            "classification": "Phishing" if pred_label == 1 else "Not phishing"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

