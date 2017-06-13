function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for i = 1:K

  % reset total and divisor
  total = zeros(1, n);
  divisor = 0;

  % Iterate over the idx assignments
    for j = 1:m

      % Debug
      % fprintf("i %f, j %f, idx(j) %f, total %f %f, divisor %f, X(j, :) %f %f\n\n",
      %   i, j, idx(j), total(1), total(2), divisor, X(j, 1), X(j, 1)
      % );
      % pause;

      % Add the entry to the total if it matches the index
      if idx(j) == i
        total = total .+ X(j, :);
        divisor = divisor + 1;
      end
    end

  % set the new means
  if divisor > 0
    centroids(i, :) = total ./ divisor;
  end

  % Debug
  % fprintf("centroids(i, :) %f %f, total %f %f, divisor %f\n\n",
  %  centroids(i, 1), centroids(i, 2), total(1), total(2), divisor
  % );
  % pause;

end

% =============================================================


end
