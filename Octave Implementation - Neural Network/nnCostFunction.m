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
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
summation = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


X = [ones(m,1), X];
upper_delta_2 = 0.0;
upper_delta_1 = 0.0;
for i=1:m
i;
a_1 = X(i,:);
%printf('The size of a_1 is');
size(a_1);
%printf('The size of Theta1 is');
size(Theta1);
Z_2 = a_1 * Theta1';
a_2 = sigmoid(Z_2);
a_2 = [ones(1,1), a_2];
Z_3 = a_2 * Theta2';
a_3 = sigmoid(Z_3);
h_x = a_3;
I = eye(num_labels);
y_1 = I(y,:);

size(y_1);

%printf('The size of Theta1 is');
size(Theta1);
%printf('The size of Theta2 is');
size(Theta2);


new_Theta1 = Theta1(:,[2:size(Theta1,2)]);
new_Theta2 = Theta2(:,[2:size(Theta2,2)]);

%size(new_Theta1);
%size(new_Theta2);

n_Theta1 = sum(sum(new_Theta1 .^ 2));

n_Theta2 = sum(sum(new_Theta2 .^ 2));

%J = sum(sum(- y_1 .* log(h_x) + (-1 * (1-y_1) .* log(1-(h_x)))))/(m) + ((lambda/(2*m))* (n_Theta1 + n_Theta2))

%J = sum(sum(- y_1(i,:) .* log(h_x(i,:)) + (-1 * (1-y_1(i,:)) .* log(1-(h_x(i,:))))))/(m)

summation = summation + sum(- y_1(i,:) .* log(h_x) - ((1-y_1(i,:)) .* log(1-(h_x)))) ;

J = (summation / m) + + ((lambda/(2*m))* (n_Theta1 + n_Theta2));


%Layer 3 lower_delta calculation

lower_delta_3 = a_3 - y_1(i,:);
%printf('The size of lower_delta_3 is');
size(lower_delta_3);

%Layer 2 calculation
b_1 = Theta2' * lower_delta_3';

%printf('The size of b_1 is');
size(b_1);

lower_delta_2 = (b_1(2:end))' .* sigmoidGradient(Z_2);

%printf('The size of lower_delta_2 is');
size(lower_delta_2);

%Calculate upper_delta

upper_delta_1 =  upper_delta_1 + (lower_delta_2' * a_1);

%printf('The size of upper_delta_1 is');
size(upper_delta_1);


upper_delta_2 =  upper_delta_2 + (lower_delta_3' * a_2);

%printf('The size of upper_delta_2 is');
size(upper_delta_2);

end


Theta1_grad(:,1) = upper_delta_1(:,1) / m;

%printf('The size of Theta1_grad is');
size(Theta1_grad);

Theta2_grad(:,1) = upper_delta_2(:,1) / m;

Theta1_grad(:,(2:end)) = (upper_delta_1(:,(2:end)) + lambda * new_Theta1)/ m;
Theta2_grad(:,(2:end)) = (upper_delta_2(:,(2:end)) + lambda * new_Theta2)/ m;



%printf('The size of Theta2_grad is');
size(Theta2_grad);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
