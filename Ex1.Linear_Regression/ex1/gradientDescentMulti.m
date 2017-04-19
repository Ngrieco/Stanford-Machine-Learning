%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

    m = length(y); % number of training examples
    p = size(X,2); % number of parameters
    
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters
        
        theta_prev = theta;
        J_history(iter) = computeCostMulti(X, y, theta);

        for i = 1:p % calculate new theta for each parameter
            theta(i) = theta(i) - alpha / m *(X*theta_prev - y)' * X(:,i);
        end  
        
    end

end
