import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle  # For saving and loading model
import math
import random

# -------------------- Helper Function for Broadcasting --------------------
def unbroadcast(grad, shape):
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

# -------------------- Vectorized Autograd Engine --------------------
class Tensor:
    def __init__(self, data, _children=(), _op=''):
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self * other.pow(-1)

    def __rtruediv__(self, other):
        return Tensor(other) * self.pow(-1)

    def pow(self, power):
        out = Tensor(self.data ** power, (self,), f'pow_{power}')
        def _backward():
            self.grad += unbroadcast(power * self.data ** (power - 1) * out.grad, self.data.shape)
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += unbroadcast(np.exp(self.data) * out.grad, self.data.shape)
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')
        def _backward():
            self.grad += unbroadcast((1 / self.data) * out.grad, self.data.shape)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backward():
            grad = out.grad
            if axis is not None:
                grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            div = self.data.size
        else:
            div = self.data.shape[axis]
        return self.sum(axis, keepdims) / div

    def matmul(self, other):
        out = Tensor(self.data.dot(other.data), (self, other), 'matmul')
        def _backward():
            # Use the transpose of 'other.data' for the left gradient
            self.grad += out.grad.dot(other.data.T)
            other.grad += self.data.T.dot(out.grad)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def transpose(self):
        out = Tensor(self.data.T, (self,), 'transpose')
        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out

    def sqrt(self):
        return self.pow(0.5)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor({self.data}, grad={self.grad})"


# -------------------- Activation & Dropout --------------------
def selu(x: Tensor):
    scale = 1.0507
    alpha = 1.67326
    data = scale * np.where(x.data > 0, x.data, alpha * (np.exp(x.data) - 1))
    out = Tensor(data, (x,), 'selu')
    def _backward():
        grad_input = np.where(x.data > 0, scale, scale * alpha * np.exp(x.data)) * out.grad
        x.grad += grad_input
    out._backward = _backward
    return out

def dropout(x: Tensor, p=0.3, training=True):
    if not training:
        return x
    mask = (np.random.rand(*x.data.shape) > p).astype(np.float32)
    data = x.data * mask / (1 - p)
    out = Tensor(data, (x,), 'dropout')
    def _backward():
        x.grad += out.grad * mask / (1 - p)
    out._backward = _backward
    return out

# -------------------- Gather for Cross Entropy Loss --------------------
def gather(t: Tensor, indices):
    # t: Tensor of shape (N, C), indices: numpy array of shape (N,)
    N = t.data.shape[0]
    gathered = t.data[np.arange(N), indices].reshape(N, 1)
    out = Tensor(gathered, (t,), 'gather')
    def _backward():
        grad = np.zeros_like(t.data)
        grad[np.arange(N), indices] = out.grad.reshape(-1)
        t.grad += grad
    out._backward = _backward
    return out

# -------------------- Cross Entropy Loss --------------------
def cross_entropy_loss(logits: Tensor, targets):
    # logits: shape (N, C); targets: numpy array of shape (N,)
    N = logits.data.shape[0]
    # Subtract max for numerical stability
    max_logits = np.max(logits.data, axis=1, keepdims=True)
    shifted = logits - Tensor(max_logits)
    exps = shifted.exp()
    sum_exps = exps.sum(axis=1, keepdims=True)
    log_sum_exps = sum_exps.log()
    log_probs = shifted - log_sum_exps
    target_log_probs = gather(log_probs, targets)
    loss = (target_log_probs * -1).sum() / N
    return loss

# -------------------- Layers --------------------
class Linear:
    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        # Weight initialization
        self.W = Tensor(np.random.randn(nout, nin) * np.sqrt(2.0 / nin))
        self.b = Tensor(np.zeros((nout,)))
        self.params = [self.W, self.b]

    def __call__(self, x: Tensor):
        # x: (N, nin) -> out: (N, nout)
        # out = x @ W.T + b
        out = x.matmul(self.W.transpose()) + self.b
        return out

class BatchNorm1d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = Tensor(np.ones((num_features,)))
        self.beta = Tensor(np.zeros((num_features,)))
        self.params = [self.gamma, self.beta]
        self.running_mean = np.zeros((num_features,), dtype=np.float32)
        self.running_var = np.ones((num_features,), dtype=np.float32)
        self.training = True

    def __call__(self, x: Tensor):
        # x: (N, num_features)
        if self.training:
            mu_val = np.mean(x.data, axis=0)
            var_val = np.mean((x.data - mu_val) ** 2, axis=0)

            mu = Tensor(mu_val)
            var = Tensor(var_val)

            self.running_mean = self.momentum * mu_val + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var_val + (1 - self.momentum) * self.running_var

            x_norm = (x - mu) / ((var + Tensor(self.eps)).pow(0.5))
        else:
            mu = Tensor(self.running_mean)
            var = Tensor(self.running_var)
            x_norm = (x - mu) / ((var + Tensor(self.eps)).pow(0.5))

        out = self.gamma * x_norm + self.beta
        return out

# -------------------- MLP Model --------------------
class MLP:
    def __init__(self, input_dim, output_dim):
        self.fc1 = Linear(input_dim, 128)
        self.bn1 = BatchNorm1d(128)
        self.fc2 = Linear(128, 256)
        self.bn2 = BatchNorm1d(256)
        self.fc3 = Linear(256, 128)
        self.bn3 = BatchNorm1d(128)
        self.fc4 = Linear(128, output_dim)

        self.dropout_p = 0.3
        self.training = True
        self.params = (
            self.fc1.params + self.bn1.params +
            self.fc2.params + self.bn2.params +
            self.fc3.params + self.bn3.params +
            self.fc4.params
        )

    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = self.bn1(x)
        x = selu(x)
        x = dropout(x, self.dropout_p, self.training)

        x = self.fc2(x)
        x = self.bn2(x)
        x = selu(x)
        x = dropout(x, self.dropout_p, self.training)

        x = self.fc3(x)
        x = self.bn3(x)
        x = selu(x)

        x = self.fc4(x)
        return x

    def __call__(self, x: Tensor):
        return self.forward(x)

# -------------------- Optimizer: AdamW --------------------
class AdamW:
    def __init__(self, params, lr=0.005, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {id(p): np.zeros_like(p.data) for p in params}
        self.v = {id(p): np.zeros_like(p.data) for p in params}

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            # Decoupled weight decay
            p.data = p.data * (1 - self.lr * self.weight_decay)

            m = self.m[id(p)]
            v = self.v[id(p)]

            m = self.betas[0] * m + (1 - self.betas[0]) * p.grad
            v = self.betas[1] * v + (1 - self.betas[1]) * (p.grad ** 2)

            m_hat = m / (1 - self.betas[0] ** self.t)
            v_hat = v / (1 - self.betas[1] ** self.t)

            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data = p.data - update

            self.m[id(p)] = m
            self.v[id(p)] = v

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

# -------------------- Learning Rate Scheduler: StepLR --------------------
class StepLR:
    def __init__(self, optimizer, step_size, gamma):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma


# -------------------- CSV Validation & Data Loading --------------------
def validate_dataframe(df):
    """
    Check that all required columns exist and are numeric, and
    that the 'label' column exists as well.
    """
    # Full column list from the screenshot
    required_cols = [
        "min", "median", "ptp", "rms", "zcr", "var",
        "skew_kurt_ratio", "crest_factor", "shape_factor",
        "skew_kurt_product", "std_min_ratio", "min_ptp_ratio", "label"
    ]

    # 1. Check presence of columns
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the CSV.")

    # 2. Check that numeric columns are numeric
    numeric_cols = [c for c in required_cols if c != "label"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric. Found dtype={df[col].dtype}.")

    # 3. Check that 'label' column is not empty
    if df["label"].isnull().any():
        raise ValueError("Some rows have a null label. Please remove or fill them.")

def load_and_preprocess_data(csv_path):
    # Read CSV
    df = pd.read_csv(csv_path)

    # Validate columns
    validate_dataframe(df)

    # Drop unwanted columns (same as PyTorch script)
    df = df.drop(columns=["rms", "skew_kurt_ratio"])

    # Drop any remaining NaNs if present
    df = df.dropna()

    # Separate label
    X_df = df.drop(columns=["label"])
    y_df = df["label"]

    # Standard scaling
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_df)

    # Label encode
    label_encoder = LabelEncoder()
    y_np = label_encoder.fit_transform(y_df)

    return X_np, y_np, label_encoder


# -------------------- Function to Save the Model --------------------
def save_model(model, filename="mlp_model.pkl"):
    """Save only the model weights (not Tensor objects) to avoid pickle issues."""
    model_weights = [p.data for p in model.params]  # Extract `.data` from Tensors
    with open(filename, "wb") as f:
        pickle.dump(model_weights, f)
    print(f"✅ Model saved successfully to {filename}.")

# -------------------- Function to Load the Model --------------------
def load_model(model, filename="mlp_model.pkl"):
    """Load model weights into an existing model instance."""
    with open(filename, "rb") as f:
        model_weights = pickle.load(f)

    for p, loaded_w in zip(model.params, model_weights):
        p.data = loaded_w  # Restore weights

    print(f"✅ Model loaded successfully from {filename}.")




# -------------------- Main Training Routine --------------------
if __name__ == '__main__':
    print("🔹 Loading Dataset...")

    # Update the CSV path as needed:
    csv_path = r"C:\Users\shaur\OneDrive\Desktop\TD_features_MLP_V2.csv"

    X_np, y_np, label_encoder = load_and_preprocess_data(csv_path)
    print("Dataset successfully validated and preprocessed.")

    # Split train/test
    X_train_np, X_test_np, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
    )

    # Wrap in Tensors
    X_train = Tensor(X_train_np)
    X_test = Tensor(X_test_np)

    # Build model
    input_dim = X_train_np.shape[1]  # should be 10 after dropping "rms" and "skew_kurt_ratio"
    output_dim = len(np.unique(y_np))  # number of classes
    model = MLP(input_dim, output_dim)

    # Hyperparameters
    epochs = 500
    best_accuracy = 0.0
    patience = 10
    patience_counter = 0

    # Optimizer & Scheduler
    optimizer = AdamW(model.params, lr=0.005, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.7)

    print("🔹 Training MLP Model...")
    for epoch in range(1, epochs + 1):
        # Training mode
        model.training = True

        # Forward pass
        logits = model(X_train)
        loss = cross_entropy_loss(logits, y_train)

        # Backward + Update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            model.training = False  # eval mode (no dropout, BN uses running stats)
            test_logits = model(X_test)
            preds = np.argmax(test_logits.data, axis=1)
            accuracy = np.mean(preds == y_test)

            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.data:.4f}, Test Accuracy: {accuracy:.4f}")

            # Save the best model during training
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
                save_model(model, filename="mlp_best_model.pkl")  # Save best model
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print("\n✅ Early Stopping Triggered. Stopping Training.")
                break

    print(f"\n🚀 Final Fine-Tuned MLP Accuracy: {best_accuracy:.4f}")
    save_model(model)  # Save final model

