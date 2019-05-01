function net = netcreate(numUnitsPerLayer, act_funcs)
%
%

L = length(numUnitsPerLayer)-2;  % number of hidden layers
if(L==0)
    init_scale = 0.01;
else
    init_scale = 1.0;
end

numUnits = numUnitsPerLayer;
net{L+2}.Z = []; % here you place minibatch of inputs
% the form of the activation functions of the hidden units and the output layer  
for layer=L+1:-1:1
    net{layer}.actfunc = str2func(act_funcs{layer});
    net{layer}.grad_actfunc = str2func(['grad_' act_funcs{layer}]);
end
% initialize the weights 
totalNumofParams = 0;  
for layer=L+1:-1:1
   if layer > 1
         net{layer}.W = randn(numUnits(layer+1), numUnits(layer))/sqrt(numUnits(layer)); 
         net{layer}.b = randn(1, numUnits(layer))/sqrt(numUnits(layer));
   else
         net{layer}.W = init_scale * randn(numUnits(layer+1), numUnits(layer)); 
         net{layer}.b = init_scale * randn(1, numUnits(layer)); 
   end
   totalNumofParams = totalNumofParams + numel(net{layer}.W) + numel(net{layer}.b);
end
net{1}.lik = act_funcs{1};
net{1}.totalNumofParams = totalNumofParams;
