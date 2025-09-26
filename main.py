import torch
from model_training import trainer
import sklearn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from model_training import collate_fn
from model_building import CNNTransformerModel
from sklearn.metrics import accuracy_score, confusion_matrix
from data_loading import AudioFeatureDataset
import json

# Number of folds
k_folds = 5

# Prepare KFold splitter (shuffling recommended)
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

with open("label_map.json", "r") as f:
            label_map = json.load(f)
        
# path_spectrogram_folder="features/spectrogram" path_mfcc_folder="features/mfcc"
dataset = AudioFeatureDataset("features/spectrogram","features/mfcc", label_map)
print("CUDA available:", torch.cuda.is_available())

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n========== Fold {fold+1}/{k_folds} ==========")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_ins = trainer("features/spectrogram","features/mfcc","label_map.json",36,512,55,fold+1,train_subset=train_subset, val_subset=val_subset)

        trained_model=train_ins.train(epochs=25)

        model_path = f"model_fold{fold+1}.pth"
        torch.save(trained_model.state_dict(), model_path)
        print(f"Model saved: {model_path}")