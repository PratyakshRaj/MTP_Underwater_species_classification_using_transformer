from torch.utils.data import Dataset
import os
import numpy as np
import torch

class AudioFeatureDataset(Dataset):
    def __init__(self, spec_root, mfcc_root, label_map, transform=None,
                 max_time=2000, min_time=64, min_mel=32, random_crop=False):
        """
        Args:
            spec_root: folder containing mel spectrogram .npy files
            mfcc_root: folder containing MFCC .npy files
            label_map: dict mapping filename -> class_idx
            max_time: max allowed time frames (axis=2)
            min_time: minimum time frames; pad if shorter
            min_mel: minimum mel bins; pad if shorter
            random_crop: whether to randomly crop long sequences
        """
        self.spec_root = spec_root
        self.mfcc_root = mfcc_root
        self.label_map = label_map
        self.files = list(label_map.keys())
        self.transform = transform
        self.max_time = max_time
        self.min_time = min_time
        self.min_mel = min_mel
        self.random_crop = random_crop

    def __len__(self):
        return len(self.files)

    def crop_pad(self, arr):
        """
        Crop/pad along time and pad along mel dimension.
        Input arr shape: (channels, mel, time)
        """
        # --- MEL dimension ---
        mel = arr.shape[1]
        if mel < self.min_mel:
            pad_mel = self.min_mel - mel
            arr = np.pad(arr, ((0,0),(0,pad_mel),(0,0)), mode="constant")

        # --- TIME dimension ---
        T = arr.shape[2]
        if T > self.max_time:
            if self.random_crop:
                start = np.random.randint(0, T - self.max_time)
                arr = arr[:, :, start:start+self.max_time]
            else:
                arr = arr[:, :, :self.max_time]
        elif T < self.min_time:
            pad_time = self.min_time - T
            arr = np.pad(arr, ((0,0),(0,0),(0,pad_time)), mode="constant")

        return arr

    def __getitem__(self, idx):
        fname = self.files[idx]
        label = self.label_map[fname]

        # assuming format species//audio.wav
        arr = fname.split("//")
        species, audio = arr[0], arr[1]
        fname = os.path.join(species, audio)

        # load features
        spec_path = os.path.join(self.spec_root, fname.replace(".wav", "_mel.npy"))
        mfcc_path = os.path.join(self.mfcc_root, fname.replace(".wav", "_mfcc.npy"))

        if not os.path.exists(spec_path) or not os.path.exists(mfcc_path):
            print("Skipping missing:", spec_path, mfcc_path)
            return None  # <<<< return None safely

        # NOW load
        spec = np.load(spec_path)
        mfcc = np.load(mfcc_path)


        # ensure proper shape
        spec = self.crop_pad(spec)
        mfcc = self.crop_pad(mfcc)

        # convert to torch
        spec = torch.tensor(spec, dtype=torch.float32)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)

        return spec, mfcc, label
