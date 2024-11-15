from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn as nn
import timm

# Define the model class
class LeukemiaModel(nn.Module):
    def __init__(self, num_classes=4):
        super(LeukemiaModel, self).__init__()
        
        # Pretrained EfficientNetB7
        self.base_model = timm.create_model('efficientnet_b7', pretrained=False, num_classes=0)
        
        # Custom layers
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(2560 * 7 * 7, 256)  # Adjust input size if needed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.dense2 = nn.Linear(256, num_classes)  # Final output layer for classification
        self.softmax = nn.Softmax(dim=1)  # Softmax for multi-class classification
        
    def forward(self, x):
        # Forward pass through the base model (EfficientNetB7)
        x = self.base_model(x)
        
        # Flatten and pass through custom layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.softmax(x)
        
        return x


# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = LeukemiaModel(num_classes=4)
model.load_state_dict(torch.load('leukemia_model.pth', weights_only=True))
model.eval()  # Set the model to evaluation mode

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to match EfficientNet input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Read and process the image
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img = transform(img).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    # Inference
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
    
    return jsonify({'prediction': int(predicted.item())})

if __name__ == '__main__':
    app.run(debug=True)
