import torch
import torchvision
import cv2
import argparse
from PIL import Image
from utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
import tkinter as tk
from tkinter import filedialog

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', default=0.965, type=float,
                    help='score threshold for discarding detection')
args = parser.parse_args()

# Initialize the model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True,
                                                           num_classes=91)

# Set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model onto the computation device and set to eval mode
model.to(device).eval()

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor()
])

# Create a GUI window and add a button to select an image
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

# Load the input image
image = Image.open(file_path).convert('RGB')

# Keep a copy of the original image for OpenCV functions and applying masks
orig_image = image.copy()

# Apply the transformation pipeline to the input image
image = transform(image)

# Add a batch dimension
image = image.unsqueeze(0).to(device)

# Obtain the segmentation masks, bounding boxes and labels
masks, boxes, labels = get_outputs(image, model, args.threshold)

# Draw the segmentation map on the original image
result = draw_segmentation_map(orig_image, masks, boxes, labels)

# Visualize the segmented image
cv2.imshow('Segmented image', result)
cv2.waitKey(0)

# Save the output image
save_path = f"../outputs/{file_path.split('/')[-1].split('.')[0]}.jpg"
cv2.imwrite(save_path, result)
