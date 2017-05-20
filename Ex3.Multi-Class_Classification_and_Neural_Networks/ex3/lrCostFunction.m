%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

function [J, grad] = lrCostFunction(theta, X, y, lambda)

    m = length(y); % number of training examples
    n = length(theta); % number of params
    J = 0;
    
    grad = zeros(size(theta));
    h_theta = sigmoid(X * theta); % vectorized predictions

    J = sum(-y .* log(h_theta) - (1 - y) .* log(1 - h_theta)) / m + ...
        lambda / (2 * m) * sum(theta(2:n).^2); % cost of function with given theta params
    
    grad(1) = (X(:,1)' * (h_theta - y)) / m;
    grad(2:n) = X(:,2:n)' * (h_theta - y) / m + lambda / m * theta(2:n);
      
    grad = grad(:);

end