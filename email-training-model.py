import opendatasets as od
od.download("https://www.kaggle.com/datasets/ethancratchley/email-phishing-dataset")

import torch
import torch.nn as nn   
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
import joblib
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc




#---Data Preprocessing---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_df = pd.read_csv('./email-phishing-dataset/email_phishing_data.csv')

data_df.dropna(inplace=True)
counts = data_df['label'].value_counts()

#---we are going to normalise the data frame (df)--- 

original_df = data_df.copy() 

for column in data_df.columns:
    if column != 'label': 
        data_df[column] = np.log1p(data_df[column])
print(data_df.head())

scaler = StandardScaler()

X = np.array(data_df.iloc[:, data_df.columns != 'label'])
Y = np.array(data_df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y) 
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, random_state=42, stratify=y_test) 

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')

smote = SMOTE(sampling_strategy=1.0,random_state=42) 
X_train, y_train = smote.fit_resample(X_train, y_train) 
counts_after = np.bincount(y_train)
pos_weight = (counts_after[0] / counts_after[1])
pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device) * 1.4

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

class dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.Y = torch.tensor(Y, dtype=torch.float32).to(device) 

    def __len__(self):
        return len(self.X) 

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx] 

#---Create objects of the dataset class for training, validation and testing datasets---

training_set = dataset(X_train, y_train)
validation_set = dataset(X_val, y_val)
testing_set = dataset(X_test, y_test)

train_loader = DataLoader(training_set, batch_size=50, shuffle=True) 
val_loader = DataLoader(validation_set, batch_size=50, shuffle=True) 
test_loader = DataLoader(testing_set, batch_size=50, shuffle=True) 

#---Define the neural net model---
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)

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

model = MyModel().to(device)

summary(model, (X.shape[1],))

#---Define the loss function and the optimizer---

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = Adam(model.parameters(), lr=1e-4) 

total_loss_train_plot = [] 
total_loss_val_plot = [] 
total_acc_train_plot = [] 
total_acc_val_plot = []

#--Training the model using loops---

for epoch in range(50): 
    
    total_acc_train = 0 
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0

    for data in train_loader:

        inputs, labels = data
        labels = labels.float() 

        prediction = model(inputs).squeeze(1)
        
        batch_loss = criterion(prediction, labels) 

        total_loss_train += batch_loss.item() 
        pred = torch.sigmoid(prediction)
        acc = ((pred>=0.5) == labels.int()).sum().item()

        total_acc_train += acc


        batch_loss.backward() 
        optimizer.step() 
        optimizer.zero_grad()

    
    

    with torch.no_grad(): 
        for data in val_loader:
            inputs, labels = data
            labels = labels.float() 
            prediction = model(inputs).squeeze(1) 
            batch_loss = criterion(prediction, labels)
            total_loss_val += batch_loss.item()
            pred = torch.sigmoid(prediction)
            acc = ((pred>=0.5) == labels.int()).sum().item()
            total_acc_val += acc 


    total_loss_train_plot.append(round(total_loss_train/training_set.__len__()*100, 4)) 
    total_loss_val_plot.append(round(total_loss_val/validation_set.__len__()*100, 4)) 
    total_acc_train_plot.append(round(total_acc_train/training_set.__len__()*100, 4)) 
    total_acc_val_plot.append(round(total_acc_val/validation_set.__len__()*100, 4)) 

    with torch.no_grad(): 
        total_loss_test = 0
        total_acc_test = 0
        for data in test_loader:
            inputs, labels = data 
            labels = labels.float() 

            prediction = model(inputs).squeeze(1) 
            batch_loss = criterion(prediction, labels) 
            total_loss_test += batch_loss.item() 
            pred = torch.sigmoid(prediction)
            acc = ((pred>=0.5) == labels.int()).sum().item() 
            total_acc_test += acc 

    print("Accuracy:", round(total_acc_test/testing_set.__len__()*100, 4), "%") 
    # Save checkpoint
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            prediction = model(inputs).squeeze(1)
            probs = torch.sigmoid(prediction)
            preds = (probs >= 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

  
    epoch_acc = round(total_acc_test / testing_set.__len__() * 100, 4)
    print(f"\nEpoch [{epoch+1}/50] Test Accuracy: {epoch_acc}%")

    # --- Confusion matrix + classification report ---
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4, target_names=["Safe (0)", "Phishing (1)"])
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    checkpoint = {
        'epoch': epoch + 1,             
        'model_state_dict': model.state_dict(),  
        'optimizer_state_dict': optimizer.state_dict(),  
        'loss': total_loss_train / len(train_loader)  
    }
    torch.save(checkpoint, 'checkpoint.pth')

#---Save the model---


torch.save(model.state_dict(), 'phishing_model.pth') 

model.eval()
all_probs, all_labels = [], []

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        prediction = model(inputs).squeeze(1)
        probs = torch.sigmoid(prediction)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


fig, axs = plt.subplots(nrows=1,ncols=2, figsize=(15, 5))

# --- Loss vs Epochs ---
axs[0].plot(total_loss_train_plot, label='Training Loss')
axs[0].plot(total_loss_val_plot, label='Validation Loss')
axs[0].set_title('Loss vs Epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_ylim(0, 2)
axs[0].legend()

# --- Accuracy vs Epochs ---
axs[1].plot(total_acc_train_plot, label='Training Accuracy')
axs[1].plot(total_acc_val_plot, label='Validation Accuracy')
axs[1].set_title('Accuracy vs Epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy (%)')
axs[1].set_ylim(0, 100)
axs[1].legend()


plt.show()



