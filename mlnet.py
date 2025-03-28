import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim

class MLNet(nn.Module):
    def __init__(self, img_size=224):
        super(MLNet, self).__init__()

        # Load pre-trained VGG16 model (feature extractor)
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features

        # Feature aggregation layers
        self.upsample_conv4 = nn.Upsample(size=(56, 56), mode='bilinear', align_corners=True)
        self.upsample_conv5 = nn.Upsample(size=(56, 56), mode='bilinear', align_corners=True)
        self.conv1x1 = nn.Conv2d(512 + 512 + 256, 64, kernel_size=1)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Feature extraction
        conv1_2 = self.features[:4](x)   # block1
        conv2_2 = self.features[4:9](conv1_2)  # block2
        conv3_3 = self.features[9:16](conv2_2)  # block3
        conv4_3 = self.features[16:23](conv3_3)  # block4
        conv5_3 = self.features[23:30](conv4_3)  # block5

        # Upsample
        conv4_upsampled = self.upsample_conv4(conv4_3)
        conv5_upsampled = self.upsample_conv5(conv5_3)

        # Concatenate features
        concat_features = torch.cat([conv3_3, conv4_upsampled, conv5_upsampled], dim=1)

        # Saliency prediction
        x = self.dropout(concat_features)
        x = F.relu(self.conv1x1(x))
        x = torch.sigmoid(self.final_conv(x))
        return x

import os
from PIL import Image

import os
from PIL import Image

class SaliencyDataset(Dataset):
    def __init__(self, image_dir, saliency_dir, transform=None):
        self.image_dir = image_dir
        self.saliency_dir = saliency_dir
        self.transform = transform

        # Define valid image extensions
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        # Ensure only image files are listed (ignore subdirectories)
        self.image_files = [
            f for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(valid_extensions)
        ]

        if len(self.image_files) == 0:
            raise ValueError(f"No valid images found in directory: {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Convert image_name to saliency map name
        base_name = os.path.splitext(image_name)[0]  # Get filename without extension
        saliency_name = base_name + "_s.png"
        saliency_path = os.path.join(self.saliency_dir, saliency_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not os.path.exists(saliency_path):
            raise FileNotFoundError(f"Saliency map not found: {saliency_path}")

        # Open images
        image = Image.open(image_path).convert("RGB")
        saliency_map = Image.open(saliency_path).convert("L")

        # Apply transforms
        if self.transform:
            image = self.transform(image)
            saliency_map = self.transform(saliency_map)

        return image, saliency_map, image_name



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_dir = "/content/drive/MyDrive/DIP/images"
saliency_dir = "/content/drive/MyDrive/DIP/TD_FixMaps"
dataset = SaliencyDataset(image_dir, saliency_dir, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLNet().to(device)

def kl_div_loss(y_true, y_pred):
    y_true_resized = F.interpolate(y_true, size=(56, 56), mode='bilinear', align_corners=True)
    y_pred_norm = y_pred / (y_pred.sum(dim=(2,3), keepdim=True) + 1e-8)
    y_true_norm = y_true_resized / (y_true_resized.sum(dim=(2,3), keepdim=True) + 1e-8)
    loss = torch.sum(y_true_norm * torch.log((y_true_norm + 1e-8) / (y_pred_norm + 1e-8)), dim=(2,3)).mean()
    return loss

optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, ground_truths, _ in dataloader:
        inputs, ground_truths = inputs.to(device), ground_truths.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = kl_div_loss(ground_truths, outputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
