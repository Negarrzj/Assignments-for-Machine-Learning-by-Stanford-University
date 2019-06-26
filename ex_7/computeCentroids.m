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

%% initializing centroind index to saparate equal index in idx and centroid_points to save actual points from X

centroid=cell(K,1);
centroid_points=cell(K,1);

for i=1:K
   % find all idx of same amount 
   s=find(idx==i);
   % save in cell 
   centroid{i}=s;
   % find actual points from X
   p=X(s,:);
   % save in cell
   centroid_points{i}=p;  
end

%% mean points of same centroid and save in centroids
for i=1:K
    for j=1:n
        centroids(i,j)=mean(centroid_points{i}(:,j));
    end
end
    
    











% =============================================================


end

