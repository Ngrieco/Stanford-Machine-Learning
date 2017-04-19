%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

function [theta] = normalEqn(X, y)

    theta = zeros(size(X, 2), 1); % size(X,2) = # of columns = # of parameters
    theta = inv(X'*X)*X'*y; % normal equation to solve for theta values
    
end
