function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Iterate over all values of X and each centroid
for i = 1:size(idx, 1)

  % Get the ith row from X
  xi = X(i, :);

  % Compare this against each centroid
  first_iter = true;
  closest = 0;
  for k = 1:K

    % Get the kth centroid value
    ck = centroids(k, :);

    % Calculate the distance
    distance = sqrt(sum((xi - ck) .^ 2));

    % debug
    % fprintf("xi %f %f, ck %f %f, closest %f, distance %f, idx %f\n\n", ...
    %  xi(1), xi(2), ck(1), ck(2), closest, distance, idx(i));
    % pause;

    % Compare the values
    if first_iter == true
      closest = distance;
      idx(i) = k;
      first_iter = false;
    elseif distance < closest
      closest = distance;
      idx(i) = k;
    end
  end
end

% =============================================================

end
