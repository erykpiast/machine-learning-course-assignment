function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
precision = 10 ^ -3;

for iter = 1:num_iters
    value = X * theta;
    diff = value - y;
    change = alpha * sum(diff .* X) / m;
    theta = theta - transpose(change);

    cost = computeCost(X, y, theta);

    J_history(iter) = cost;
  
    if (cost < precision)
        break;
    end;
end

end
