%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps priceou get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, priceou will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization
clear ; close all; clc

%% ================ Part 1: Feature Normalization ================

fprintf('Loading data ...\n');

% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
price = data(:, 3);
m = length(price);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], price = $%.0f \n', [X(1:10,:) price(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);


% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = .1;
num_iters = 50;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, price, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Displaprice gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house

predict = [1 (1650 - mu(1))/sigma(1) (3 - mu(2))/sigma(2)];
price = predict * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n\n');

pause;


%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
price = data(:, 3);
m = length(price);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, price);

% Displaprice normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta)
;
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house

predict = [1 1650 3];
price = predict * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);


