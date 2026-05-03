# --------------------------------------------------
# Load library and data
# --------------------------------------------------

import numpy as np

fname = "assign1_data.csv"
data = np.genfromtxt(fname, dtype="float", delimiter=",", skip_header=1)
X, y = data[:, :-1], data[:, -1].astype(int)
# FIRST 400 rows to training; the rest to testing
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]

# --------------------------------------------------
# Layer: Dense (fully connected): z = X @ W + b
# --------------------------------------------------


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        """Weights - N(0,1)*0.01; biases = 0.0."""
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.inputs = None
        self.z = None
        self.dweights = None
        self.dbiases = None
        self.dinputs = None

        # setting up all the parameters and
        # variables needed for a dense layer to perform forward and backward propagation in a neural network.

    def forward(self, inputs):
        """Compute z and cache inputs for backprop."""
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        # produces the layer’s output (before activation) and caches the inputs for use during the backward pass.

    def backward(self, dz):
        """Backprop: dW, db, and dinputs."""
        self.dweights = np.dot(self.inputs.T, dz)
        self.dbiases = np.sum(dz, axis=0, keepdims=True)
        self.dinputs = np.dot(dz, self.weights.T)
        # computes the gradients of the loss with respect to the layer’s weights, biases, and inputs.

# --------------------------------------------------
# Activation: ReLU and Softmax
# --------------------------------------------------


class ReLu:
    """ReLU: a = max(0, z)."""

    def __init__(self):
        self.z = None
        self.activity = None
        self.dz = None
    # sets up placeholders to store the input and gradient values needed for the ReLU activation function.

    def forward(self, z):
        self.z = z
        self.activity = np.maximum(0.0, z)

    def backward(self, dactivity):
        self.dz = dactivity.copy()
        self.dz[self.z <= 0] = 0.0

    # this class implements ReLU in my network and correctly propogates gradients during training.


class Softmax:
    """Row-wise softmax → probabilities that sum to 1."""

    def __init__(self):
        self.probs = None
        self.dz = None

    def forward(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.probs = e_z / e_z.sum(axis=1, keepdims=True)
        return self.probs

    def backward(self, dprobs):
        """Jacobian-vector product."""
        self.dz = np.empty_like(dprobs)
        for i, (prob, dprob) in enumerate(zip(self.probs, dprobs)):
            prob = prob.reshape(-1, 1)
            jacobian = np.diagflat(prob) - np.dot(prob, prob.T)
            self.dz[i] = np.dot(jacobian, dprob)

    # implements the softmax activation function and its backward pass using the Jacobian matrix.

# --------------------------------------------------
# Loss: Cross-Entropy (with one-hot targets)
# --------------------------------------------------


class CrossEntropyLoss:
    def __init__(self):
        self.dprobs = None

    def forward(self, probs, oh_y_true):
        """Mean CE; clip to avoid log(0)."""
        probs_clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
        loss = -np.sum(oh_y_true * np.log(probs_clipped), axis=1)
        return float(loss.mean(axis=0))

    def backward(self, probs, oh_y_true):
        """Gradient wrt probs, normalized by batch size."""
        batch_sz = probs.shape[0]
        self.dprobs = -oh_y_true / probs
        self.dprobs = self.dprobs / batch_sz

    # computes the cross-entropy loss and its gradient with respect to the predicted probabilities.

# --------------------------------------------------
# Optimizer: SGD (per-layer update)
# --------------------------------------------------


class SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights = layer.weights - self.learning_rate * layer.dweights
        layer.biases = layer.biases - self.learning_rate * layer.dbiases

    # This class applies the SGD update rule to each layer,
    # moving weights and biases in the direction that reduces the loss.

# --------------------------------------------------
# Helper functions
# --------------------------------------------------


def predictions(probs):
    """Argmax over classes."""
    return np.argmax(probs, axis=1)


def accuracy(y_preds, y_true):
    """Mean correct predictions."""
    return np.mean(y_preds == y_true)

# --------------------------------------------------
# Forward / Backward passes (whole network)
# --------------------------------------------------


def forward_pass(X, y_true, oh_y_true):
    """Single forward pass: returns (probs, loss) as per task text."""
    dense1.forward(X)
    activation1.forward(dense1.z)
    dense2.forward(activation1.activity)
    activation2.forward(dense2.z)
    dense3.forward(activation2.activity)
    probs = output_activation.forward(dense3.z)
    loss = crossentropy.forward(probs, oh_y_true)
    return probs, loss
    # This function orchestrates the forward propagation through the entire network,
    # returning the final predicted probabilities and the computed loss.


def backward_pass(probs, y_true, oh_y_true):
    """Single backward pass: CE - Softmax - Dense3 - ReLU - Dense2 - ReLU - Dense1."""
    crossentropy.backward(probs, oh_y_true)              # dL/dprobs
    output_activation.backward(crossentropy.dprobs)      # dL/dz3
    dense3.backward(output_activation.dz)                # dL/d(a2)
    activation2.backward(dense3.dinputs)                 # dL/dz2
    dense2.backward(activation2.dz)                      # dL/d(a1)
    activation1.backward(dense2.dinputs)                 # dL/dz1
    dense1.backward(activation1.dz)                      # dL/dX
    # This function manages the backward propagation through the network,
    # computing gradients at each layer in the reverse order of the forward pass.

# forward pass computes outputs and loss by passing data through the network
# backward pass computes gradients by propagating the loss gradient back through the network

# --------------------------------------------------
# Initialize network, hyperparameters, and train
# --------------------------------------------------


if __name__ == "__main__":
    n_features = X_train.shape[1]  # expected 3
    n_class = 3

    # Build exact architecture: 3-4-8-3
    dense1 = DenseLayer(n_inputs=n_features, n_neurons=4)
    activation1 = ReLu()
    dense2 = DenseLayer(n_inputs=4, n_neurons=8)
    activation2 = ReLu()
    dense3 = DenseLayer(n_inputs=8, n_neurons=n_class)
    output_activation = Softmax()

    crossentropy = CrossEntropyLoss()
    # per skeleton default; 0.5–1.0 both work
    optimizer = SGD(learning_rate=1.0)

    epochs = 10
    batch_sz = 32
    n_train = X_train.shape[0]
    n_batch = (n_train + batch_sz - 1) // batch_sz

    for epoch in range(1, epochs + 1):
        print("epoch:", epoch)

        # (optional but standard) shuffle each epoch
        perm = np.random.permutation(n_train)
        X_train, y_train = X_train[perm], y_train[perm]

        # mini-batches
        for batch_i in range(n_batch):
            s = batch_i * batch_sz
            e = min(s + batch_sz, n_train)
            Xb = X_train[s:e]
            yb = y_train[s:e]
            oh_yb = np.eye(n_class)[yb]              # one-hot

            # Forward
            probs_b, loss_b = forward_pass(Xb, yb, oh_yb)

            # Metrics
            preds_b = predictions(probs_b)
            acc_b = accuracy(preds_b, yb)
            print(
                f"  batch {batch_i+1:02d}/{n_batch:02d} | loss={loss_b:.4f} | acc={acc_b:.4f}")

            # Backward
            backward_pass(probs_b, yb, oh_yb)

            # Update (output - hidden2 - hidden1)
            optimizer.update_params(dense3)
            optimizer.update_params(dense2)
            optimizer.update_params(dense1)

# --------------------------------------------------
# Test evaluation (print test accuracy)
# --------------------------------------------------

    oh_y_test = np.eye(n_class)[y_test]
    probs_test, test_loss = forward_pass(X_test, y_test, oh_y_test)
    test_preds = predictions(probs_test)
    test_acc = accuracy(test_preds, y_test)
    print(f"\nTEST | loss={test_loss:.4f} | acc={test_acc:.4f}")
