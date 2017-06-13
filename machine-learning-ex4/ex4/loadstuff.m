%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

% Load Training Data
load('ex4data1.mat');
m = size(X, 1);

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');

% Unroll parameters
nn_params = [Theta1(:) ; Theta2(:)];

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Create the y_matrix
y_matrix = zeros(m, num_labels);
for i = 1:m
  y_matrix(i, y(i)) = 1;
end

% Feedforward
a1 = [ones(m, 1) X]; % add 1s to the training set
z2 = a1 * Theta1'; % Used in part 2 below
a2 = sigmoid(z2); % first hidden layer
a2 = [ones(size(a2,1), 1) a2]; % add 1s to the output of the first hidden layer
z3 = a2 * Theta2';
a3 = sigmoid(z3); % output layer
