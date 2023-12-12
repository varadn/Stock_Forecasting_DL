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

    


    





















































































# import numpy as np
# from Modules.rnn_utils.utilities import Tanh, Softmax, CrossEntropyLoss

# # Recurrent Neural Network Implementation
# class RNN_Scratch:
#     def __init__(self, input_dim, output_dim, hidden_dim):
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim

#         params = self._initialize_parameters(
#                 input_dim, output_dim, hidden_dim
#         )
        
#         self.Wya, self.Wax, self.Waa, self.by, self.b = params
#         self.softmax = Softmax()
#         self.oparams = None
 
#     # Computes Forward Propagation
#     def forward(self, input_X):
#         self.input_X = input_X

#         # Initialize Layers
#         self.layers_tanh = [Tanh() for x in input_X]
#         hidden = np.zeros((self.hidden_dim , 1))
#         self.hidden_list = [hidden]
#         self.y_preds = []

#         # Forward Propagate using tanh
#         for input_x, layer_tanh in zip(input_X, self.layers_tanh):
#             input_tanh = np.dot(self.Wax, input_x) + np.dot(self.Waa, hidden) + self.b
#             hidden = layer_tanh.forward(input_tanh)
#             self.hidden_list.append(hidden)

#             input_softmax = np.dot(self.Wya, hidden) + self.by
#             y_pred = self.softmax.forward(input_softmax)
#             self.y_preds.append(y_pred)

#         return self.y_preds
 
#     # Computes Loss
#     def loss(self, Y):
#         self.Y = Y
#         self.layers_loss = [CrossEntropyLoss() for y in self.Y]
#         cost = 0
        
#         # compute loss through each layer
#         for y_pred, y, layer in zip(self.y_preds, self.Y, self.layers_loss):
#             cost += layer.forward(y_pred, y)
        
#         return cost
 
#     # Computes Backward Propagation
#     def backward(self):  
#         gradients = self._define_gradients()
#         self.dWax, self.dWaa, self.dWya, self.db, self.dby, dhidden_next = gradients

#         for index, layer_loss in reversed(list(enumerate(self.layers_loss))):
#             dy = layer_loss.backward()

#             # hidden actual
#             hidden = self.hidden_list[index + 1]
#             hidden_prev = self.hidden_list[index]

#             # gradients y
#             self.dWya += np.dot(dy, hidden.T)
#             self.dby += dy
#             dhidden = np.dot(self.Wya.T, dy) + dhidden_next
    
#             # gradients a
#             dtanh = self.layers_tanh[index].backward(dhidden)
#             self.db += dtanh
#             self.dWax += np.dot(dtanh, self.input_X[index].T)
#             self.dWaa += np.dot(dtanh, hidden_prev.T)
#             dhidden_next = np.dot(self.Waa.T, dtanh)
 
#     # Clips gradients to avoid exploding gradient
#     def clip(self, clip_value):
#         for gradient in [self.dWax, self.dWaa, self.dWya, self.db, self.dby]:
#             np.clip(gradient, -clip_value, clip_value, out=gradient)
 
#     # Updates params of the model
#     def optimize(self, method):
#         weights = [self.Wya, self.Wax, self.Waa, self.by, self.b]
#         gradients = [self.dWya, self.dWax, self.dWaa, self.dby, self.db]

#         weights, self.oparams = method.optim(weights, gradients, self.oparams)
#         self.Wya, self.Wax, self.Waa, self.by, self.b = weights

#     def generate_names(self, index_to_character):
#         letter = None
#         indexes = list(index_to_character.keys())

#         letter_x = np.zeros((self.input_dim, 1))
#         name = []

#         # similar to forward propagation.
#         layer_tanh = Tanh()
#         hidden = np.zeros((self.hidden_dim , 1))

#         while letter != '\n' and len(name)<15:

#             input_tanh = np.dot(self.Wax, letter_x) + np.dot(self.Waa, hidden) + self.b
#             hidden = layer_tanh.forward(input_tanh)

#             input_softmax = np.dot(self.Wya, hidden) + self.by
#             y_pred = self.softmax.forward(input_softmax)

#             index = np.random.choice(indexes, p=y_pred.ravel())
#             letter = index_to_character[index]

#             name.append(letter)

#             letter_x = np.zeros((self.input_dim, 1))
#             letter_x[index] = 1

#         return "".join(name)


#     def _initialize_parameters(self, input_dim, output_dim, hidden_dim):
#         den = np.sqrt(hidden_dim)

#         weights_y = np.random.randn(output_dim, hidden_dim) / den
#         bias_y = np.zeros((output_dim, 1))

#         weights_ax = np.random.randn(hidden_dim, input_dim) / den
#         weights_aa = np.random.randn(hidden_dim, hidden_dim) / den
#         bias = np.zeros((hidden_dim, 1))

#         return weights_y, weights_ax, weights_aa, bias_y, bias


#     def _define_gradients(self):
#         dWax = np.zeros_like(self.Wax)
#         dWaa = np.zeros_like(self.Waa)
#         dWya = np.zeros_like(self.Wya)

#         db = np.zeros_like(self.b)
#         dby = np.zeros_like(self.by)

#         da_next = np.zeros_like(self.hidden_list[0])

#         return dWax, dWaa, dWya, db, dby, da_next