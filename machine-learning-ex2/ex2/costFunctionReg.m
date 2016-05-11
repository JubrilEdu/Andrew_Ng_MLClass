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



h =  sigmoid(X * theta );
first_sum = transpose((-(y))) * log(h);
second_sum = transpose((1-y))* log(1-h);
unregularised_cost = (first_sum-second_sum)*(1/m);
theta(1) = 0;
regularized_cost = (lambda / (2 * m))*(theta'*theta);
J_one = (lambda/m)*theta;
J =  unregularised_cost+regularized_cost;
grad = ((1/m)*transpose(X)*(h - y)) + J_one;


% =============================================================

end
