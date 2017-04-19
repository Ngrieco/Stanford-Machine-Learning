%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

function [X_norm, mu, sigma] = featureNormalize(X)

    X_norm = X;
    mu = zeros(1, size(X, 2)); % size(X,2) = # of columns
    sigma = zeros(1, size(X, 2));
    normalize = zeros(size(X,1), size(X, 2)); % size(X,1) = # of rows

    for i = 1:size(X,2)
        mu(i) = mean(X_norm(:,i));
        sigma(i) = std(X_norm(:,i));
        normalize(:,i) = (X_norm(:,i)-mu(i))./sigma(i);
    end;
    
    X_norm = normalize;

end

