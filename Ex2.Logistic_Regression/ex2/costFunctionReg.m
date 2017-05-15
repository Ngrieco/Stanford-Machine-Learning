%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

function [J, grad] = costFunctionReg(theta, X, y, lambda)

    m = length(y); % number of training examples
    n = length(theta);
    J = 0;
    grad = zeros(size(theta));

    h_theta = sigmoid(X * theta);
    
    J = sum(-y .* log(h_theta) - (1-y) .* log(1 - h_theta)) / m + ...
        lambda / (2 * m) * sum(theta(2:end).^2);
    
    grad(1) = sum((h_theta-y).*X(:,1)) / m;
    for i = 2:n
        grad(i) = sum((h_theta-y).*X(:,i)) / m + lambda / m * theta(i);
    end

end
