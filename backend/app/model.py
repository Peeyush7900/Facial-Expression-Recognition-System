import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Emotion mapping consistent with typical FER-2013 structures
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

class EmotionCNN(nn.Module):
    """
    A 3-stage Convolutional Neural Network optimized for 48x48 grayscale face images.
    """
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        # Stage 1: Input 1x48x48 -> 64x24x24
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)
        
        # Stage 2: 64x24x24 -> 128x12x12
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)
        
        # Stage 3: 128x12x12 -> 256x6x6 (This is the target layer for Grad-CAM)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.25)
        
        # Fully Connected Classifier: 256*6*6 = 9216 -> 512 -> 7
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, len(EMOTIONS))
        
    def forward(self, x):
        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Stage 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Stage 3 (conv6 is our last conv layer, target for Grad-CAM)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Flatten and Classify
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)
        x = self.fc2(x)
        return x

class GradCAM:
    """
    Grad-CAM class to extract class activation maps.
    Registers a single forward hook to capture activations and backpropagates gradients to them.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook = None
        self.register_hooks()
        
    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            # Register a hook on the output tensor directly to capture its gradients during backward pass
            def backward_hook(grad):
                self.gradients = grad
            output.register_hook(backward_hook)
            
        self.hook = self.target_layer.register_forward_hook(forward_hook)
        
    def __call__(self, x: torch.Tensor, class_idx: int = None):
        """
        Executes Grad-CAM for a given input tensor and class index.
        Returns:
            - cam: 2D numpy array representing the normalized heatmap
            - output: Raw model output logits
            - class_idx: The class index utilized for Grad-CAM
        """
        self.model.eval()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        if self.gradients is None or self.activations is None:
            # Fallback to zeros if hooks didn't trigger correctly
            return np.zeros((48, 48), dtype=np.float32), output, class_idx
            
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Mean gradients across channels (global average pooling)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to cam
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam, output, class_idx
        
    def remove_hooks(self):
        if self.hook is not None:
            self.hook.remove()

def get_attention_regions(cam: np.ndarray):
    """
    Given a 48x48 Grad-CAM activation heatmap, computes relative attention weights
    across eyes, eyebrows, nose, and mouth regions.
    """
    if cam is None or cam.size == 0:
        return {"eyebrows": 0.25, "eyes": 0.25, "nose": 0.25, "mouth": 0.25}, "nose"
        
    # Resize heatmap to 48x48 if it's from conv layers with smaller resolution
    if cam.shape != (48, 48):
        import cv2
        cam = cv2.resize(cam, (48, 48))
        
    # Bounding boxes on the 48x48 pixel grid
    eyebrows_val = float(np.mean(cam[8:18, 6:42]))
    eyes_val = float(np.mean(cam[14:24, 7:41]))
    nose_val = float(np.mean(cam[20:33, 15:33]))
    mouth_val = float(np.mean(cam[31:43, 9:39]))
    
    regions = {
        "eyebrows": eyebrows_val,
        "eyes": eyes_val,
        "nose": nose_val,
        "mouth": mouth_val
    }
    
    total = sum(regions.values())
    if total > 0:
        regions = {k: v / total for k, v in regions.items()}
    else:
        regions = {k: 0.25 for k in regions.keys()}
        
    primary = max(regions, key=regions.get)
    return regions, primary

