clear; 
addpath act_functs;

randn('seed',1);
rand('seed',0);

[X,T,Xtest,TtestTrue] = loadMnist();
[N, D] = size(X);
K = size(T,2);

% Create the neural net
numUnitsPerHiddenLayer = [300];  % units per hidden layer  (the length of this vector is the number of hidden layers)
inputDim = D; 
outputDim = K;
numUnitsPerLayer = [outputDim numUnitsPerHiddenLayer inputDim];  % all units from output (left) to the input (right)
% possible choices for the activations functions: 'relu' 'softmax' 'lin' 'cos' 'sigmoid' 'tanh' 'softplus' 
act_funcs = {'softmax', 'relu'}; 
net = netcreate(numUnitsPerLayer, act_funcs); 

iters = 250000;
etaW = 0.5/N; 
eta0 = 0.005; 
TT = 1;     
kappa0 = 0.1;
ReduceRateEvery = 10000;
ReduceRateBy = 0.9;
numBatch = 100; 
likfactor = N/numBatch;


for layer=1:length(net)-1
    Gt_net{layer}.W = 0.01*ones(size( net{layer}.W ));
    Gt_net{layer}.b = 0.01*ones(size( net{layer}.b ));
end

cost = zeros(1,iters); 
st = 1; 
perm = randperm(N);
tic;
for it=1:iters 
%      
    % take the next minibatch 
    [block, st, perm] = takeNextBatch(N, numBatch, st, perm);
     
    % cost and gradients for the minibatch
    [out, grad_W, grad_b] = netcost(X(block,:), T(block,:), net);
    cost(it) = likfactor*out;  
        
    % RMSprop Update of the parameters
    kappa = kappa0;
    if(it==1)
        kappa = 1;
    end
    for layer=length(net)-1:-1:1
        Gt_net{layer}.W = kappa*(grad_W{layer}.^2) + (1-kappa)*Gt_net{layer}.W;
        Gt_net{layer}.b = kappa*(grad_b{layer}.^2) + (1-kappa)*Gt_net{layer}.b;
        
        net{layer}.W = net{layer}.W + eta0*grad_W{layer}./(TT+sqrt(  Gt_net{layer}.W ));
        net{layer}.b = net{layer}.b + eta0*grad_b{layer}./(TT+sqrt(  Gt_net{layer}.b ));
        
        %net{layer}.W = net{layer}.W + etaW*grad_W{layer};
        %net{layer}.b = net{layer}.b + etaW*grad_b{layer};
    end
        
    if mod(it,100) == 0    
        fprintf('Iter=%d, cost=%f\n', it, cost(it));
    end
    
    if mod(it,ReduceRateEvery) == 0
        etaW = ReduceRateBy*etaW;
        eta0 = ReduceRateBy*eta0;
    end
%   
end
timetaken = toc;

% Test the model 
net = netforward(net, Xtest); 
[~, Ttest] = max(net{1}.Z,[],2); 
[~, Ttrue] = max(TtestTrue,[],2); 
err = length(find(Ttest~=Ttrue))/10000;

disp(['The error of the method is: ' num2str(err)])

