%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

function [J, grad] = costFunction(theta, X, y)
    
    m = length(y); % number of training examples
    n = size(theta); % number of parameters
    J = 0;
    grad = zeros(n);
    
    h_theta = sigmoid(X * theta); % M x 1 vector holding predictions for each of X's inputs
                                % each row = sigmoid(theta0 + theta1*X1 + ... + ?thetan*Xn)
    
    J = sum(-y .* log(h_theta) - (1-y) .* log(1 - h_theta)) / m; % cost of function with given theta params
    for i = 1:n
        grad(i) = sum((h_theta-y).*X(:,i)) / m;
    end
    
    
end