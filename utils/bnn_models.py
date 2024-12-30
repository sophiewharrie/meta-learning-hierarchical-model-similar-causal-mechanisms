import torch
import torch.nn as nn

from custom_lstm import CustomLSTM

class LinearNNModel(nn.Module):
    """Creates a neural network model with n_layers linear layers and a final output layer for binary classification.

    Parameters:
    - in_features (int): the number of features (predictors) in the input
    - n_layers (int): the number of linear layers to use in the BNN
    - out_features_list (list of int): the output size of each linear layer (length of list = n_layers)
    """
    def __init__(self, in_features, n_layers, out_features_list):
        super(LinearNNModel, self).__init__()
        self.linear_layers = []

        # add the specified number of linear layers
        for i in range(n_layers):
            out_features = out_features_list[i]
            self.linear_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
        
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.relu = nn.ReLU()
        self.final_layer = nn.Linear(in_features, 2)

    def forward(self, tabular_data, sequence_data, return_embeddings=False):
        x = tabular_data
        for layer in self.linear_layers:
            x = self.relu(layer(x))
        logits = self.final_layer(x)

        if return_embeddings: return logits, x
        else: return logits


def linear_nn_model(in_features, n_layers, out_features_list):
    model = LinearNNModel(in_features, n_layers, out_features_list)
    return model


class SequenceNNModel(nn.Module):
    """
    Creates a neural network model that combines sequence (longitudinal) data and tabular data for binary classification.

    Parameters:
    - num_features_longitudinal (int): Number of input features for sequence (longitudinal) data.
    - num_features_tabular (int): Number of input features for tabular data.
    - hidden_dim_longitudinal (int): Number of hidden units in the LSTM layer.
    - hidden_dim_tabular (int): Number of hidden units in the fully connected layer for tabular data.
    """
    def __init__(self, num_features_longitudinal, num_features_tabular, hidden_dim_longitudinal, hidden_dim_tabular):
        super(SequenceNNModel, self).__init__()
        
        # LSTM for sequence data
        self.lstm = CustomLSTM(input_size=num_features_longitudinal, 
                            hidden_size=hidden_dim_longitudinal, 
                            batch_first=True)
        
        # fully connected layer for tabular data
        self.fc_tabular = nn.Linear(num_features_tabular, hidden_dim_tabular)
        
        # output layer
        self.fc_out = nn.Linear(hidden_dim_longitudinal + hidden_dim_tabular, 2)

        # activation function
        self.act = nn.ReLU()

    def forward(self, tabular_data, sequence_data, return_embeddings=False):
        # LSTM forward pass for sequence data
        # - use the output from the last time step
        # (because it acts as a summary of all previous elements in the sequence)
        lstm_out, _ = self.lstm(sequence_data) # sequence_data is of shape (num_patients, num_years, num_endpoints)
        lstm_out = lstm_out[:, -1, :]
        
        # fully connected forward pass for tabular data
        tabular_out = self.fc_tabular(tabular_data)
        tabular_out = self.act(tabular_out)
        
        # concatenate LSTM and tabular outputs
        combined_out = torch.cat((lstm_out, tabular_out), dim=1)
        
        # output layer for classification
        logits = self.fc_out(combined_out)
        
        if return_embeddings: return logits, combined_out, lstm_out
        else: return logits


def sequence_nn_model(num_features_longitudinal, num_features_tabular, hidden_dim_longitudinal, hidden_dim_tabular):
    model = SequenceNNModel(num_features_longitudinal, num_features_tabular, hidden_dim_longitudinal, hidden_dim_tabular)
    return model
