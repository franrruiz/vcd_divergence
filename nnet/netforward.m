function net = netforward(net, X)

% number of hidden layers (i.e. excluding the output and input layer) 
L = length(net)-2;

if nargin == 2
   % input layer: N is the minibatch size
   net{L+2}.Z = X;
end

%N = size(net{L+2}.Z, 1);

% forward pass in the neural network
for layer=L+1:-1:1
   net{layer}.A = bsxfun(@plus, net{layer+1}.Z*net{layer}.W, net{layer}.b);  % N x M_ell matrix  
   net{layer}.Z = net{layer}.actfunc(net{layer}.A);
end
