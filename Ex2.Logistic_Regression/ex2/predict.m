%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

function p = predict(theta, X)

    m = size(X, 1); % Number of training examples
    p = zeros(m, 1); % admit/not admit predictions for each input

    h_theta = sigmoid(X*theta); % calculate predictions

    admitted = find(h_theta >= .5); % indices of predictions >= 0.5
    notAdmitted = find(h_theta < .5); % indices of predictions < 0.5

    h_theta(admitted) = 1; 
    h_theta(notAdmitted) = 0;

    p = h_theta;


end
