%% Initialization
clear ; close all; clc

%% Setup the parameters 
input_layer_size  = 6483;  
hidden_layer_size = 350;   
num_labels = 4;           


fprintf('Loading Data ...\n')

X = load('train_file.txt');
X = X(:,[2:end]);
y= load('train_y.txt');
size(X)
size(y)



fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];




lambda = 3;

debug_J  = nnCostFunction(initial_nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCost Function J(theta): %f ' ...
         '\n\n\n'], debug_J);

fprintf('\nTraining Neural Network... \n')


options = optimset('MaxIter', 10);


lambda = 1;


costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);


[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));



fprintf('\nVisualizing Neural Network... \n')



fprintf('\nProgram paused. Press enter to continue.\n');



X = load('test_file.txt');
X = X(:,[2:end]);
y= load('test_y.txt');
size(X)
size(y) 
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


