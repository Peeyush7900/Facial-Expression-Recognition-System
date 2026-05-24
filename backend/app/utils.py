import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def base64_to_cv2(b64_str: str) -> np.ndarray:
    """
    Converts a base64 encoded image string (with or without headers) to a CV2 RGB image.
    """
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_data = base64.b64decode(b64_str)
    img_pil = Image.open(BytesIO(img_data)).convert("RGB")
    return np.array(img_pil)

def cv2_to_base64(img_rgb: np.ndarray, format: str = "JPEG") -> str:
    """
    Converts an RGB image array to a base64 encoded data URI.
    """
    img_pil = Image.fromarray(img_rgb)
    buffered = BytesIO()
    img_pil.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

def generate_gradcam_overlay(crop_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blends a 2D float heatmap (0..1) with a cropped RGB face image.
    Uses the JET colormap.
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return np.zeros((48, 48, 3), dtype=np.uint8)
        
    h, w, _ = crop_rgb.shape
    
    # Resize heatmap to match the crop size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Scale heatmap to 0-255
    heatmap_255 = np.uint8(255 * heatmap_resized)
    
    # Apply JET colormap (returns BGR)
    heatmap_color = cv2.applyColorMap(heatmap_255, cv2.COLORMAP_JET)
    
    # Convert colormap BGR to RGB
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend crop with heatmap: output = crop * (1 - alpha) + heatmap_color * alpha
    blended = cv2.addWeighted(crop_rgb, 1.0 - alpha, heatmap_color_rgb, alpha, 0)
    
    return blended
