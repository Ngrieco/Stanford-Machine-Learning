%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1);
        
    for iter = 1:num_iters

        J_history(iter) = computeCost(X, y, theta); % store cost after each iteration
        
        h = X*theta; % h is Mx1 vector
        
        theta0 = theta(1) - sum( (h - y) .* X(:,1) ) * alpha / m; % calculate new thetas first
        theta1 = theta(2) - sum( (h - y) .* X(:,2) ) * alpha / m;
            
        theta = [theta0; theta1]; % then, update both values

    end

end