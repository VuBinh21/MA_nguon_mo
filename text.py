# Mạng nơ-ron feedforward 2 lớp (1 hidden layer) học phép toán XOR. 
# Mạng dùng sigmoid, gradient descent (batch) và backpropagation
import math

# ---- hàm tiện ích ----
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(y):
    # y = sigmoid(x)
    return y * (1.0 - y)

def mse_loss(targets, outputs):
    # mean squared error
    return sum((t - o) ** 2 for t, o in zip(targets, outputs)) / len(targets)

# ---- lớp mạng 2 lớp (input -> hidden -> output) ----
class SimpleNeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output, lr=0.5):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr = lr

        # khởi tạo trọng số (lists) + bias
        # weights_input_hidden: n_hidden x n_input
        self.w_ih = [[random.uniform(-1, 1) for _ in range(n_input)] for _ in range(n_hidden)]
        # bias cho hidden
        self.b_h = [random.uniform(-1, 1) for _ in range(n_hidden)]

        # weights_hidden_output: n_output x n_hidden
        self.w_ho = [[random.uniform(-1, 1) for _ in range(n_hidden)] for _ in range(n_output)]
        # bias cho output
        self.b_o = [random.uniform(-1, 1) for _ in range(n_output)]

    def forward(self, x):
        # x: list length n_input
        # hidden activations
        self.hidden_net = []
        self.hidden_out = []
        for i in range(self.n_hidden):
            net = sum(self.w_ih[i][j] * x[j] for j in range(self.n_input)) + self.b_h[i]
            out = sigmoid(net)
            self.hidden_net.append(net)
            self.hidden_out.append(out)

        # output layer
        self.output_net = []
        self.output_out = []
        for k in range(self.n_output):
            net = sum(self.w_ho[k][i] * self.hidden_out[i] for i in range(self.n_hidden)) + self.b_o[k]
            out = sigmoid(net)
            self.output_net.append(net)
            self.output_out.append(out)

        return self.output_out[:]  # copy

    def backward(self, x, target):
        # x: input list
        # target: output list
        # forward pass already computed, gradients computed now

        # output layer deltas
        delta_o = [0.0] * self.n_output
        for k in range(self.n_output):
            error = target[k] - self.output_out[k]
            delta_o[k] = error * sigmoid_derivative(self.output_out[k])

        # hidden layer deltas
        delta_h = [0.0] * self.n_hidden
        for i in range(self.n_hidden):
            # sum of downstream contributions
            downstream = sum(self.w_ho[k][i] * delta_o[k] for k in range(self.n_output))
            delta_h[i] = downstream * sigmoid_derivative(self.hidden_out[i])

        # update weights hidden->output
        for k in range(self.n_output):
            for i in range(self.n_hidden):
                grad = delta_o[k] * self.hidden_out[i]
                self.w_ho[k][i] += self.lr * grad
            # update bias for output k
            self.b_o[k] += self.lr * delta_o[k]

        # update weights input->hidden
        for i in range(self.n_hidden):
            for j in range(self.n_input):
                grad = delta_h[i] * x[j]
                self.w_ih[i][j] += self.lr * grad
            # update bias for hidden i
            self.b_h[i] += self.lr * delta_h[i]

    def train(self, data_inputs, data_targets, epochs=10000, verbose=True):
        # data_inputs: list of input lists
        # data_targets: list of target lists
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            # batch gradient descent: accumulate update per example (we update per-example here)
            for x, t in zip(data_inputs, data_targets):
                outputs = self.forward(x)
                total_loss += mse_loss(t, outputs)
                self.backward(x, t)
            avg_loss = total_loss / len(data_inputs)
            if verbose and (epoch % (epochs // 10) == 0 or epoch <= 20):
                print(f"Epoch {epoch:5d} / {epochs}   loss={avg_loss:.6f}")

    def predict(self, x):
        out = self.forward(x)
        return out

# ---- ví dụ: học XOR ----
if __name__ == "__main__":
    # dữ liệu XOR
    inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    targets = [
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ]

    nn = SimpleNeuralNetwork(n_input=2, n_hidden=2, n_output=1, lr=0.5)
    print("Đang train mạng cho XOR ...")
    nn.train(inputs, targets, epochs=10000, verbose=True)

    print("\nKết quả dự đoán sau huấn luyện:")
    for x, t in zip(inputs, targets):
        y = nn.predict(x)[0]
        print(f"Input={x}  Target={t[0]}  Pred={y:.4f}  Rounded={round(y)}")
