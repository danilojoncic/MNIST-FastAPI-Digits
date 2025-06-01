# Neural Networks From Scratch

![image](https://github.com/user-attachments/assets/feb5c442-08ef-4257-85da-2a4e97c2b8cb)


A complete implementation of neural networks using only NumPy, featuring a web-based digit recognition interface powered by FastAPI.

## 🧠 Overview

This project demonstrates building neural networks from the ground up without machine learning frameworks. It includes:

- **Custom neural network implementation** using pure NumPy
- **MNIST digit classification** with hand-drawn input capability
- **Interactive web interface** for real-time predictions
- **RESTful API** for model inference
- **Dockerized deployment** for easy setup

## 🏗️ Architecture

The neural network consists of:
- **Input Layer**: 784 neurons (28×28 pixel images)
- **Hidden Layer 1**: 256 neurons with ReLU activation
- **Hidden Layer 2**: 128 neurons with ReLU activation  
- **Output Layer**: 10 neurons with Softmax activation (digits 0-9)

## 📁 Project Structure

```
├── starter.py              # Core neural network implementation
├── runner.py               # Model training script
├── main.py                 # FastAPI server and web interface
├── model_loader.py         # Model loading and inference utilities
├── mnist.pkl.gz           # MNIST dataset (compressed pickle)
├── mnist_model_params.pkl  # Pre-trained model parameters
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
└── README.md              # This file
```

## 🚀 Quick Start

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd neural-networks-from-scratch
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

3. **Access the application**
   - Open your browser to `http://localhost:8080/paint-screen`
   - Draw a digit and click "Predict" to see the neural network's prediction

### Manual Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the FastAPI server**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. **Access the web interface**
   - Navigate to `http://localhost:8000/paint-screen`

## 🎯 Features

### Web Interface
- **Interactive drawing canvas** for digit input
- **Real-time 28×28 preview** showing processed image
- **Instant predictions** with confidence scores
- **Image download** functionality for debugging

### Neural Network Implementation
- **Dense layers** with customizable neurons
- **Activation functions**: ReLU, Softmax
- **Optimizers**: SGD, Adam, RMSprop, Adagrad
- **Loss functions**: Categorical Cross-entropy
- **Forward and backward propagation** from scratch

### API Endpoints
- `GET /paint-screen` - Web interface for drawing digits
- `POST /predict` - Image prediction endpoint
- `POST /debug-image` - Debug processed images
- `GET /health-check` - Service health status

## 🔧 Model Training

To retrain the model with your own parameters:

```bash
python runner.py
```

This will:
1. Load the MNIST dataset from `mnist.pkl.gz`
2. Train the neural network using runner.py
3. Save trained parameters to `mnist_model_params.pkl`

## 📊 Model Performance

The pre-trained model achieves:
- **Training accuracy**: ~98%
- **Validation accuracy**: ~96%
- **Inference time**: <50ms per prediction

## 🛠️ Technical Details

### Image Processing Pipeline
1. **Canvas drawing** captured at 280×280 resolution
2. **Downsampling** to 28×28 pixels with anti-aliasing
3. **Color inversion** (white background → black, black drawing → white)
4. **Normalization** to 0-1 pixel value range
5. **Flattening** to 784-element vector for neural network input

### Neural Network Components
- **Layer_Dense**: Fully connected layers with weight initialization
- **Activation_ReLU**: Rectified Linear Unit activation
- **Activation_Softmax**: Softmax activation for probability distribution
- **Loss_CategoricalCrossentropy**: Cross-entropy loss calculation
- **Optimizers**: Various gradient descent variants with momentum

## 🔍 Debugging

Use the debug endpoint to inspect processed images:

```bash
curl -X POST "http://localhost:8000/debug-image" \
     -H "Content-Type: application/json" \
     -d '{"image": "data:image/png;base64,..."}'
```

This returns image statistics and saves the processed image for inspection.

## 📋 Requirements

- nnfs 0.5.1
- fastapi 0.115.12
- uvicorn 0.34.2
- numpy 2.2.6
- Pillow 10.3.0
- python-multipart 0.0.9
- MarkupSafe 3.0.2
- jinja2 3.1.6

See `requirements.txt` for complete dependency list.

## 🐳 Docker Configuration

The project includes multi-stage Docker builds for:
- **Development**: Hot-reload enabled
- **Production**: Optimized image size and performance

```yaml
# Development
docker-compose up

# Production
docker-compose -f docker-compose.prod.yml up
```

## 🙏 Acknowledgments

- **Neural Networks from Scratch** books
