function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

% The multiplicative steps
steps = [0.01 0.03 0.1 0.3 1 3 10 30];

% Iterate across each pair of candidate values to find the optimum C and sigma values
lowest_error = 10000;
for i = 1:length(steps)
  C_i = steps(i);
  for j = 1:length(steps)
    sigma_j = steps(j);

    % Train the model
    fprintf("\n\nTraining with C = %f, sigma = %f\n", C_i, sigma_j);
    model = svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_j));

    % Run the predictions and calculate the error
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));

    fprintf("Error %f, C was %f, sigma was %f\n", error, C, sigma);

    if (error < lowest_error)
      C = C_i;
      sigma = sigma_j;
      lowest_error = error;
    end

    fprintf("Lowest error %f, C now %f, sigma now %f\n", lowest_error, C, sigma);

  end
end

% =========================================================================

end
