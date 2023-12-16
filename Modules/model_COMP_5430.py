import numpy as np

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim, lr):
        # Create layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Create weights
        self.input_hidden_weight = np.random.uniform(0,1,(self.hidden_dim, self.input_dim)) / 2
        self.hidden_hidden_weight = np.random.uniform(0,1,(self.hidden_dim, self.hidden_dim)) / 2
        self.output_hidden_weight = np.random.uniform(0,1,(self.output_dim, self.hidden_dim)) / 2
        # Learning Rate
        self.learning_rate = lr

    # Derivative of tanh(x)
    def deriv_tanh(self,x):
        return 1 - np.square(np.tanh(x))

    def forward(self, input):
        # Initialize Hidden States as a zero matrix
        hidden_states = []
        hidden_states.append(np.zeros((self.hidden_dim, 1)))
        # Forward pass
        for i in range(input.shape[0]):
            # get next hidden step
            next_hidden_state = np.tanh(np.dot(self.input_hidden_weight, input[[i]].T) + np.dot(self.hidden_hidden_weight, hidden_states[-1]))
            # save next hidden step
            hidden_states.append(next_hidden_state)
        # Output from Forward pass
        hidden_output = np.dot(self.output_hidden_weight, hidden_states[-1])
        return hidden_states, hidden_output
    
    # Update weights
    def update_weights(self, hid_inp_w, hid_hid_w, out_hid_w):
        self.input_hidden_weight -= self. learning_rate * hid_inp_w
        self.hidden_hidden_weight -= self.learning_rate * hid_hid_w
        self.output_hidden_weight -= self.learning_rate * out_hid_w

    # Using Mean Squared Error as Loss Function
    def loss(self, actual, predicted):
        return np.mean(np.square(actual - predicted))
    
    # Backpropagation
    def backprop(self, input, target, hidden_states, hidden_output):
        # Calculate Loss
        error = self.loss(target, hidden_output)
        # Init weights to shape of original weights
        in_hid_w = np.zeros_like(self.input_hidden_weight)
        hid_hid_w = np.zeros_like(self.hidden_hidden_weight)
        out_hid_w = np.zeros_like(self.output_hidden_weight)
        # Gradient error w.r.t. hidden_hidden_w
        grad_err = np.dot(self.output_hidden_weight.T, error)
        # Gradient tanh w.r.t. hidden_hidden
        gradient_hidden_states = grad_err * self.deriv_tanh(hidden_states[-1])

        # Iter backwards through each timestep
        for i in reversed(range(input.shape[0])):
            # Add grad's to weights
            hid_hid_w += np.dot(gradient_hidden_states, hidden_states[i-1].T)
            in_hid_w += np.dot(gradient_hidden_states, input[[i-1]])

        # Add gradient to output weight
        out_hid_w += np.dot((hidden_output- target), hidden_states[-1].T)
        # Update weights
        self.update_weights(in_hid_w, hid_hid_w, out_hid_w)

    # Train input data
    def train(self, input, target):
            epochs = 10
            # For each epoch
            for epoch in range(epochs):
                # list outputs
                if epoch == epochs - 1:
                    train_output_list = []
                for i in range(input.shape[0]):
                    hidden_states, hidden_output = self.forward(input[i])
                    # to measure training accuracy
                    if epoch == epochs - 1:
                        # save outputs
                        train_output_list.append(hidden_output.tolist()[0])
                    # backprop
                    self.backprop(input[i], target[i], hidden_states, hidden_output)
            # Transpose outputs
            train_output_list = np.array(train_output_list).T[0]
            # return trained outputs
            return train_output_list

    # Test Output data
    def test(self, input):
        # list to save output
        test_out_list = []
        # forward step
        for i in range(input.shape[0]):
            hidden_states, hidden_output = self.forward(input[i])
            # save outputs
            test_out_list.append(hidden_output.tolist()[0])
        # conver to array and transpose outputs
        test_out_list = np.array(test_out_list).T[0]
        return test_out_list