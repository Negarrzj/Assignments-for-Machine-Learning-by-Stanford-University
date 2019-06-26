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
%% d= distance .... find distance of each point from centroids 

for i=1:size(X,1)
    d=zeros(1,K);
    clear M
    clear ind
    for j=1:K
        %%save them in distance dd
        dd = X(i,:)-centroids(j,:);
        dd2 = dd.^2;
            
        %d(j)=sqrt(power(X(i,1)-centroids(j,1),2)+ power(X(i,2)-centroids(j,2),2));
        d(j)=sqrt(sum(dd2));
       
    end
    %%find min distance
    [M,ind]=min(d);
    %%save index of that centroid that made min distance in idx
    idx(i)=ind;
end
        






% =============================================================

end

