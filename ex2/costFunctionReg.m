function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


grad1=(1/m)*(X(:,1))'*(sigmoid(X*theta)-y); % for first column without reg

% for the rest of matrix
grad2=(1/m)*(X(:,[2:end]))'*(sigmoid(X*theta)-y)+(lambda/m)*theta(2:end); 
grad_T=[grad1 grad2']; % combine grad1 and grad2
grad=grad_T'; % transpose grad

% compute cost function
J=(1/m)*((-y'*log(sigmoid(X*theta)))-(ones(m,1)-y)'*log(1-sigmoid(X*theta)))+(lambda/(2*m))*(sum(theta(2:end).^2));




% =============================================================

end
