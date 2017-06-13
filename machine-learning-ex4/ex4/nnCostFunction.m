function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

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

% Calculate cost
J = (1 / m) * sum(sum(-y_matrix .* log(a3) + -(1 - y_matrix) .* log(1 - a3))); % not sure why I need to use sum twice (returns a scalar first time around)

% Add the regularisation elements
Theta1_nb = Theta1;
Theta1_nb(:,1) = 0; % remove the bias column
Theta2_nb = Theta2;
Theta2_nb(:,1) = 0; % remove the bias column
J = J + (lambda / (2 * m) * (sum(sum(Theta1_nb .^ 2)) + sum(sum(Theta2_nb .^ 2))));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%

% Cost in the output layer
delta3 = a3 .- y_matrix;

% Cost in the hidden layer
delta2 = delta3 * Theta2;
delta2(:,1)=[]; % remove the bias units column
delta2 = delta2 .* sigmoidGradient(z2);

% Calculate product and sum of all errors
DELTA1 = delta2' * a1; % Don't really understand this step
DELTA2 = delta3' * a2; % Don't really understand this step

% Update the gradients
Theta1_grad = (1 / m) * DELTA1; % Theta1 is 25 x 401
Theta2_grad = (1 / m) * DELTA2; % Theta2 is 10 x 26


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Calculate the regularization
reg1 = Theta1 .* (lambda / m);
reg1(:, 1) = 0;
reg2 = Theta2 .* (lambda / m);
reg2(:, 1) = 0;

% Apply to
Theta1_grad = Theta1_grad + reg1;
Theta2_grad = Theta2_grad + reg2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
