%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

function J = computeCost(X, y, theta)

    X = X(:,2);  % X = population vector; ignore X(:,1) [ones vector]
    theta0 = theta(1);
    theta1 = theta(2);

    m = length(y); % number of training examples
    J = 0; % running cost

    for i = 1:m
        h = theta0 + theta1.*X(i); %
        cost = (h - y(i)).^2; % cost for each data entry
        J = J + cost;
    end;

    J = J/(2*m);                                                                           

end

