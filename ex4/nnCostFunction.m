function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); % 25 x 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % 10 x 26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m, 1) X]; % m x 401
z2 = a1 * Theta1'; % m x 25
a2 = [ones(size(z2, 1), 1) sigmoid(z2)]; % m * 26
z3 = a2 * Theta2'; % m x 10
a3 = sigmoid(z3); % m x 10
h = a3; % m x 10
positive_cost = log(h); # m x 10
negative_cost = log(1 - h); # m x 10

% smart trick to translate y vector to matrix with binary vectors as rows
% (y = 1 -> [1 0 0 0 0 0 0 0 0 0], y = 2 -> [0 1 0 0 0 0 0 0 0 0] etc.)
%
% How does it work?
% Passing vector of numbers as index of matrix (M(vec, :)) will choose
% rows at all given positions; so for M([1 4 3 3], :), rows 1 and 4 are taken
% from matrix M, row 2 is omitted and row 3 is taken two times.
labeled_y = eye(num_labels)(y, :); % m x 10

d3 = a3 - labeled_y; % m x 10
grad_2 = d3' * a2; % 10 x 26
Theta2_grad = ((1 / m) * grad_2) + ((lambda / m) * ...
                                    [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)]);

d2 = (d3 * Theta2) .* sigmoidGradient([ones(size(z2, 1), 1) z2]); % m * 26
% we have to cut off the first row (j = 0) and make the matrix equal in size
% with Theta1
grad_1 = (d2' * a1)(2:end, :); % 25 x 401
Theta1_grad = ((1 / m) * grad_1) + ((lambda / m) * ...
                                    [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)]);

thetas_wo_0 = [Theta1(:, 2:end)(:); Theta2(:, 2:end)(:)];
theta_penalty = (lambda * sum(thetas_wo_0.^2)) / (2 * m);

J = (sum((
           (-labeled_y .* positive_cost) - ...
           ((1 - labeled_y) .* negative_cost)
          )(:)) / m) + theta_penalty;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
