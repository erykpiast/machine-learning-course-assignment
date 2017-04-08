function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

function y = h(x, theta)
    y = theta(1) + (theta(2) * x);
end

function normalized = normalizeData(X)
    normalized = zeros(size(X));
    for i = 1:size(X)(2)
        range = max(X(:, i)) - min(X(:, i));
        m = mean(X(:, i));
        if range > 0
            normalized(:, i) = (X(:, i) - m) / range;
        else
            normalized(:, i) = ones(length(X), 1) * 0.5;
        end
    end
end

% n_X = [X(:, 1), normalizeData(X(:, 2:end))]
n_X = X;
n_y = y;
for iter = 1:num_iters
    value = n_X * theta;
    diff = value - n_y;
    std_dev = sum(diff .* n_X) / m;
    theta = theta - transpose(alpha * std_dev);

    J_history(iter) = computeCost(n_X, n_y, theta);
end

end
