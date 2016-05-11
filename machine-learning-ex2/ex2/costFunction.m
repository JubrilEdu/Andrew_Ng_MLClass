function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

h =  sigmoid(X * theta );
first_sum = transpose((-(y))) * log(h);
second_sum = transpose((1-y))* log(1-h);
unregularised_cost = (first_sum-second_sum)*(1/m);
%third_sum = theta.*theta;
%regularized_cost = (lambda / (2 * m))*transpose(third_sum);
J =  unregularised_cost;
grad = (1/m)*transpose((h - y))*X;

% =============================================================

end
