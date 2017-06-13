function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of parameters

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Calculate hypothesis results
h = X * theta;

% Calculate cost (vectorised version)
J = (1 / (2 * m)) * sum((h .- y) .^ 2);

% Regularize the results for cost
J = J + (lambda / (2 * m) * sum(theta([2:n]) .^ 2));

% Calculate grad (vectorised version)
grad = (X' * (h - y)) ./ m;

% Regularised element for grad
temp = theta;
temp(1) = 0;
grad = grad .+ (temp .* (lambda / m));

% =========================================================================

grad = grad(:);

end
