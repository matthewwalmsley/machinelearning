function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this value correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

% Get the range of possibilities
range = size(X, 1);

% Pick a random sample for each centroid (should consider making these unique?)
for i = 1:K
  centroids(i, :) = X(round(rand(1) * range), :);
end

% =============================================================

end
