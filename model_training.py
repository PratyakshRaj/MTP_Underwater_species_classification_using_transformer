import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from data_loading import AudioFeatureDataset
from model_building import CNNTransformerModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
import os
from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch):
    specs, mfccs, labels = zip(*batch)

    # ---- Spectrogram ----
    specs = [s.squeeze(0) if s.dim() == 3 else s for s in specs]  # make sure [mel, T]
    specs = [s.permute(1, 0) for s in specs]                      # → [T, mel]
    spec_lengths = [s.shape[0] for s in specs]
    specs_padded = pad_sequence(specs, batch_first=True)          # [B, T_max, mel]
    specs_padded = specs_padded.permute(0, 2, 1).unsqueeze(1)     # [B, 1, mel, T_max]
    spec_mask = torch.arange(specs_padded.shape[-1])[None, :] < torch.tensor(spec_lengths)[:, None]

    # ---- MFCC ----
    mfccs = [m.squeeze(0) if m.dim() == 3 else m for m in mfccs]  # make sure [features, T]
    mfccs = [m.permute(1, 0) for m in mfccs]                      # → [T, features]
    mfcc_lengths = [m.shape[0] for m in mfccs]
    mfccs_padded = pad_sequence(mfccs, batch_first=True)          # [B, T_max, features]
    mfcc_mask = torch.arange(mfccs_padded.shape[1])[None, :] < torch.tensor(mfcc_lengths)[:, None]
    mfccs_padded = mfccs_padded.permute(0, 2, 1)                   # [B, features, T_max]

    labels = torch.tensor(labels)

    return specs_padded, mfccs_padded, labels, spec_mask, mfcc_mask




class trainer:
    def __init__(self,path_spec_folder,path_mfcc_folder,label_map_path,d_mfcc_features, d_model, num_classes, fold,train_subset=None, val_subset=None):
        with open(label_map_path, "r") as f:
            self.label_map = json.load(f)

        # If subsets are provided (K-Fold), wrap them in DataLoaders
        if train_subset is not None and val_subset is not None:
            self.train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, collate_fn=collate_fn)
            self.val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, collate_fn=collate_fn)    
        else:
            # Otherwise just use the full dataset (your original code)
            full_dataset = AudioFeatureDataset(path_spec_folder, path_mfcc_folder, self.label_map)
            self.train_loader = DataLoader(full_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
            self.val_loader = None  # no validation in this case




        # path_spectrogram_folder="features/spectrogram" path_mfcc_folder="features/mfcc"
        #self.train_dataset = AudioFeatureDataset(path_spec_folder, path_mfcc_folder, self.label_map)
        #self.train_loader = DataLoader(self.train_dataset, batch_size=4, shuffle=True,collate_fn=collate_fn)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # d_mfcc_features=36, d_model=512, num_classes=55
        self.model = CNNTransformerModel(d_mfcc_features, d_model, num_classes).to(self.device)
        self.fold=fold
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4)

    def train(self,epochs):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_idx,(spec, mfcc, labels, spec_mask, mfcc_mask) in enumerate(self.train_loader):
                spec, mfcc, labels = spec.to(self.device), mfcc.to(self.device), labels.to(self.device)
                spec_mask, mfcc_mask = spec_mask.to(self.device), mfcc_mask.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(spec, mfcc, spec_mask=spec_mask, mfcc_mask=mfcc_mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if batch_idx % 5 == 0:  # print every 2 batches
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}")


            print(f"Epoch {epoch+1}: Loss = {total_loss/len(self.train_loader):.4f}")
             
            # ---- Evaluate both train and validation ----
        train_acc = self.evaluate_loader(self.train_loader, name="Train",base_dir="train_val_results")
        if self.val_loader:
            val_acc = self.evaluate_loader(self.val_loader, name="Val",base_dir="train_val_results")            
    
        return self.model  

    def evaluate_loader(self,loader,name="Val",base_dir="train_val_results"):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for spec, mfcc, labels, spec_mask, mfcc_mask in loader:
                spec, mfcc, labels = spec.to(self.device), mfcc.to(self.device), labels.to(self.device)
                spec_mask, mfcc_mask = spec_mask.to(self.device), mfcc_mask.to(self.device)

                outputs = self.model(spec, mfcc, spec_mask=spec_mask, mfcc_mask=mfcc_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
              
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        avg_loss = total_loss / len(loader)

        print(f"{name} Loss = {avg_loss:.4f}, {name} Acc = {acc:.4f}, {name} F1 = {f1:.4f}")

        results = {
                "name": name,
                "loss": avg_loss,
                "accuracy": acc,
                "f1": f1
            }

        if name == "Val":  # confusion matrix only for validation
          
            cm = confusion_matrix(all_labels, all_preds).tolist()
            results["confusion_matrix"] = cm
            print(f"[Fold {self.fold}] Validation Confusion Matrix:\n", cm)
        
        
        fold_dir = os.path.join(base_dir, f"fold_{self.fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        save_file = os.path.join(fold_dir, f"{name.lower()}.json")

        with open(save_file, "a") as f:  # append mode so we don’t overwrite previous results
            f.write(json.dumps(results) + "\n")    

        




