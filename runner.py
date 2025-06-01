import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import pickle
import gzip
import time
from starter import Layer_Dense, Activation_ReLU, Model, Activation_Softmax, Loss_CategoricalCrossentropy, \
    Optimizer_Adam, Activation_Softmax_Loss_CategoricalCrossentropy

nnfs.init()


class MNISTRunner:
    def __init__(self):
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_and_prepare_data()

        self.model = Model()
        self.model.add(Layer_Dense(784, 256))  # Layer 0: Dense layer
        self.model.add(Activation_ReLU())  # Layer 1: ReLU activation
        self.model.add(Layer_Dense(256, 128))  # Layer 2: Dense layer
        self.model.add(Activation_ReLU())  # Layer 3: ReLU activation
        self.model.add(Layer_Dense(128, 10))  # Layer 4: Dense layer
        self.model.add(Activation_Softmax())
        self.loss_function = Loss_CategoricalCrossentropy()
        self.optimizer = Optimizer_Adam(
            learning_rate=0.001,  # Lower initial rate
            decay=1e-4,
            epsilon=1e-7,
            beta_1=0.9,
            beta_2=0.999
        )

        self.batch_size = 64
        self.epochs = 20
        self.print_every = 100

    def load_and_prepare_data(self):
        (train_X, train_y), (val_X, val_y), (test_X, test_y) = load_data()

        train_X = train_X / 255.0
        val_X = val_X / 255.0
        test_X = test_X / 255.0

        if len(train_y.shape) > 1:
            train_y = np.argmax(train_y, axis=1)
        if len(val_y.shape) > 1:
            val_y = np.argmax(val_y, axis=1)
        if len(test_y.shape) > 1:
            test_y = np.argmax(test_y, axis=1)

        return train_X, train_y, val_X, val_y, test_X, test_y

    def train(self):
        print("Starting training...")
        train_steps = len(self.X_train) // self.batch_size

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")

            indices = np.arange(len(self.X_train))
            np.random.shuffle(indices)
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]

            epoch_loss = 0
            correct_predictions = 0

            start_time = time.time()

            for step in range(train_steps):
                batch_start = step * self.batch_size
                batch_end = (step + 1) * self.batch_size
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]

                output = self.model.forward(X_batch)

                loss = self.loss_function.calculate(output, y_batch)
                epoch_loss += loss

                predictions = np.argmax(output, axis=1)
                correct_predictions += np.sum(predictions == y_batch)

                self.loss_function.backward(output, y_batch)
                self.model.backward(self.loss_function.dinputs)

                self.optimizer.pre_update_params()
                for layer in self.model.layers:
                    if isinstance(layer, Layer_Dense):
                        self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if step % self.print_every == 0 or step == train_steps - 1:
                    accuracy = correct_predictions / ((step + 1) * self.batch_size)
                    print(f"Step {step + 1}/{train_steps} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

            epoch_loss /= train_steps
            epoch_accuracy = correct_predictions / (train_steps * self.batch_size)
            epoch_time = time.time() - start_time

            val_loss, val_accuracy = self.validate()

            print(
                f"Summary - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f} - Time: {epoch_time:.2f}s")

    def validate(self):
        output = self.model.forward(self.X_val)

        loss = self.loss_function.calculate(output, self.y_val)
        predictions = np.argmax(output, axis=1)
        accuracy = np.mean(predictions == self.y_val)

        return loss, accuracy

    def test(self):
        print("\nTesting model...")
        output = self.model.forward(self.X_test)

        loss = self.loss_function.calculate(output, self.y_test)
        predictions = np.argmax(output, axis=1)
        accuracy = np.mean(predictions == self.y_test)

        print(f"Test Loss: {loss:.4f} - Test Accuracy: {accuracy:.4f}")
        return loss, accuracy

    def save_model(self, path='mnist_model_params.pkl'):
        self.model.save_parameters(path)

    def load_model(self, path='mnist_model_params.pkl'):
        self.model.load_parameters(path)


def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()

    train_X, train_y = training_data
    val_X, val_y = validation_data
    test_X, test_y = test_data

    return (train_X, train_y), (val_X, val_y), (test_X, test_y)


if __name__ == "__main__":
    runner = MNISTRunner()

    runner.train()
    runner.test()
    runner.save_model()
    print("\nTraining complete! Model parameters saved.")