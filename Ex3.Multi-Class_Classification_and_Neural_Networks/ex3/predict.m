%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

function p = predict(Theta1, Theta2, X)

    m = size(X, 1);
    num_labels = size(Theta2, 1);
    p = zeros(size(X, 1), 1);

    a1 = [ones(m,1) X];
    a2 = sigmoid(a1 * Theta1');
    a2 = [ones(m,1) a2];
    a3 = sigmoid(a2 * Theta2');
    
    [M, p] = max(a3, [], 2);
    
end