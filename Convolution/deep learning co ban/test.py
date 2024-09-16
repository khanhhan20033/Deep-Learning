import cv2
import torch
import albumentations as A
from transformers import TrOCRProcessor
from matplotlib import pyplot as plt

# Define your config class or dictionary for device settings
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# Initialize the processor (using a pre-trained model for TrOCR)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# Path to your image
path = 'train_OCR/TestImage/Screenshot 2024-03-28 161307.png'

# Load the image using OpenCV
img = cv2.imread(str(path))

# Define a default transformation if none is provided
transform = A.Compose([
            A.Rotate(5, border_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.Perspective((0.01, 0.05), pad_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.InvertImg(p=0.05),
            A.OneOf([
                A.Downscale(0.25, 0.5, interpolation=cv2.INTER_LINEAR),
                A.Downscale(0.25, 0.5, interpolation=cv2.INTER_NEAREST),
            ], p=0.1),
            A.Blur(p=0.2),
            A.Sharpen(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise((50, 200), p=0.3),
            A.ImageCompression(quality_lower=1, quality_upper=30, p=0.1),
            A.ToGray(always_apply=True),
        ])

# Apply the transformation (convert image to grayscale)
img = transform(image=img)['image']

# Convert the image from grayscale to RGB for visualization
# If the image is already single-channel (grayscale), we need to handle it for matplotlib
if len(img.shape) == 2:  # If the image is grayscale, convert it to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
else:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.imshow(img_rgb)
plt.title("Transformed Image")
plt.axis("off")  # Hide axes for a cleaner visualization
plt.show()

# Convert the image to tensor format using TrOCRProcessor
pixel_values = processor(img, return_tensors="pt").pixel_values

# Move the tensor to the correct device
pixel_values = pixel_values.squeeze().to(config.DEVICE)

# Print the tensor shape to ensure it was processed correctly
print(pixel_values.shape)
