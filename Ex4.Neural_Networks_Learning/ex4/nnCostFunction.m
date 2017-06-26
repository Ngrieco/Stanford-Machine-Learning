%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%% Cost Function

    % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    % for our 2 layer neural network
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
                 
    m = size(X, 1); % # of training examples
    k = size(Theta2,1); % # of outputs

    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));

    a1 = [ones(m,1), X];
    z2 = a1 * Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(m,1) a2];
    z3 = a2 * Theta2';
    htheta = sigmoid(z3);

    for i = 1:m
        hthetak = htheta(i,:)';
        yk = zeros(1, num_labels);
        yk(y(i)) = 1;
        J = J + -yk * log(hthetak) - (1 - yk) * log(1 - hthetak);   %% check why I had to transpose yk and hthetak
    end
    
    theta1 = Theta1(:,2:end);
    theta2 = Theta2(:,2:end);
    J = J / m + lambda / (2*m) * (sum(sum(theta1.^2)) + sum(sum(theta2.^2)));


    
%% Backpropagation

    y_matrix = zeros(size(htheta));
    for l = 1:k
        y_matrix(:,l) = (y == l); % turn y into matrix form for easier calculations
    end
   
    delta_3 = htheta - y_matrix; %5000 x 10
    delta_2 = delta_3 * theta2 .*  sigmoidGradient(z2); %5000 x 25
    
    Theta1_grad = Theta1_grad + delta_2' * a1 / m; %25 x 401
    Theta2_grad = Theta2_grad + delta_3' * a2 / m; %10 x 26

    t1 = [zeros(size(theta1,1),1), theta1]; % 25 x 401
    t2 = [zeros(size(theta2,1),1), theta2]; %10 x 26
    
    Theta1_grad = Theta1_grad + lambda / m * t1;
    Theta2_grad = Theta2_grad + lambda / m * t2;
    
    
    % Unroll gradients 
    grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
