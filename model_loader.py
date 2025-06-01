import numpy as np
import pickle
from starter import Layer_Dense, Activation_ReLU, Activation_Softmax, Model


class MNISTPredictor:
    def __init__(self, model_path='mnist_model_params.pkl'):
        self.model = self._load_model(model_path)

    def _load_model(self, path):
        model = Model()
        model.add(Layer_Dense(784, 256))
        model.add(Activation_ReLU())
        model.add(Layer_Dense(256, 128))
        model.add(Activation_ReLU())
        model.add(Layer_Dense(128, 10))
        model.add(Activation_Softmax())

        with open(path, 'rb') as f:
            params = pickle.load(f)

        for layer, layer_params in zip(model.layers, params):
            if layer_params is not None:
                layer.weights = layer_params['weights']
                layer.biases = layer_params['biases']

        return model

    def preprocess(self, image_data):
        """
        Preprocess input image data for the model

        Args:
            image_data: numpy array or list of pixel values
                       Should be 784 elements (28x28 flattened) or (28, 28) shape

        Returns:
            Preprocessed image ready for model input
        """
        image = np.array(image_data, dtype=np.float32)

        # Handle different input shapes
        if image.shape == (28, 28):
            image = image.flatten()
        elif image.size == 784:
            image = image.flatten()
        else:
            raise ValueError(f"Image must be 28x28 pixels. Got shape: {image.shape}")

        image = image.reshape(1, 784)

        if image.max() > 1.0:
            image = image / 255.0

        return image

    def predict(self, image_data):

        try:
            processed_image = self.preprocess(image_data)
            output = self.model.forward(processed_image)
            prediction = int(np.argmax(output))
            confidence = float(np.max(output))
            probabilities = output.flatten().tolist()

            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probabilities,
                'raw_output_shape': output.shape
            }

        except Exception as e:
            return {
                'error': str(e),
                'prediction': -1,
                'confidence': 0.0,
                'probabilities': [0.0] * 10
            }

    def debug_model_info(self):

        print("Model Architecture:")
        print("-" * 40)
        for i, layer in enumerate(self.model.layers):
            layer_type = type(layer).__name__
            if hasattr(layer, 'weights'):
                print(f"Layer {i}: {layer_type} - Input: {layer.weights.shape[0]}, Output: {layer.weights.shape[1]}")
            else:
                print(f"Layer {i}: {layer_type}")
        print("-" * 40)
