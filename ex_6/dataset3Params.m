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

%% initialization
%%series of possible C and sigma
C_series=[0.01,0.03,0.1,0.3,1,3,10,30];
sigma_series=[0.01,0.03,0.1,0.3,1,3,10,30];

%%creat an array to collect error of using diffrenet C and sigma
error_collection=cell(0);
%%index for error collection
index=0;

%% run SVM model and find error for each C and sigma from series 
for i=1:size(C_series,2)
    for j=1:size(sigma_series,2)
        C=C_series(i);
        sigma=sigma_series(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        pre_error=mean(double(predictions ~= yval));
        index=index+1;
        error_collection{index}=[C,sigma,pre_error];
        errerValues(index)=pre_error;
    end
end 

%% find min of error and corresponding C and sigma
[m,indx]= min(errerValues);
C=error_collection{indx}(1,1);
sigma=error_collection{indx}(1,2);

% =========================================================================

end
