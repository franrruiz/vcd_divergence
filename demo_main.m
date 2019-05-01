function demo_main(method_id, data_id, model_id, BurnIters, SamplingIters)
% 
% Perform amortized inference on latent variable models by minimizing the
% VCD divergence
% 
% INPUTS:
%  + method_id: The method of choice
%    (1=VCD; 2=VCD(no controlVar); 3=standardKL; 4=Hoffman's)
%  + data_id: The dataset
%    (1=MNIST; 2=Fashion-MNIST)
%  + model_id: Indicate the model
%    (1=BernoulliVAE; 2=Gaussian MF; 3=Poisson MF; 4=Logistic MF)
%  + BurnIters: Number of burn iterations in the HMC procedure
%  + SamplingIters: Number of sampling iterations in the HMC procedure
% 


%% Specify setting
if(method_id==1)
    method_name = 'VCD';
elseif(method_id==2)
    method_name = 'VCD_noControlVar';
elseif(method_id==3)
    method_name = 'standardKL';
elseif(method_id==4)
    method_name = 'Hoffman';
else
    error(['Unknown method id: ' num2str(method_id)]);
end

if(data_id==1)
    data_name = 'mnist';
elseif(data_id==2)
    data_name = 'fashionmnist';
else
    error(['Unknown data id: ' num2str(data_id)]);
end

%% Seed
randn('seed',1);
rand('seed',1);

%% Add folders to path
addpath nnet; 
addpath mcmc;
addpath aux;
addpath compute;
addpath logdensities;

%Output folder
outdir = ['./out/burn' num2str(BurnIters) '_sampling' num2str(SamplingIters) '/'];
if(~isdir(outdir))
    mkdir(outdir);
end
if(model_id==1)
    outName = [data_name 'VAE_' method_name];
elseif(model_id==2)
    outName = [data_name 'gaussMF_' method_name];
elseif(model_id==3)
    outName = [data_name 'poissMF_' method_name];
elseif(model_id==4)
    outName = [data_name 'logisticMF_' method_name];
else
    error(['Unknown model id: ' num2str(model_id)]);
end

%% Parameters
if(model_id==1)
    dim_z = 10;                 % dimensionality of the latent z
elseif(model_id==2)
    dim_z = 50;                 % number of latent factors
	variance_lik = 0.01;        % variance of the observations
elseif(model_id==3)
    dim_z = 50;                 % number of latent factors
elseif(model_id==4)
    dim_z = 50;                 % number of latent factors
else
    error(['Unknown model id: ' num2str(model_id)]);
end

% Batch size
batchsize = 100;

% Control variates
flag_control_variate = 0;
if(method_id==1)
    flag_control_variate = 1;    % use local control variates?
    decay_control_variate = 0.9; % exponential decay parameter
    control_iters_wait = 3000;   % number of initial iterations with shared control variates
end

% Stepsize parameters
rhoModelParams = 0.0005;
rhotheta = 0.0005;
rhosigma = 0.00025;
ReducedBy = 0.9; 
ReducedEvery = 15000;

% RMSprop parameters
TT = 1;  
kappa0 = 0.1;

%% Load the data
flag_binarize_data = 0;
flag_normalize_data = 0;
if(model_id==1)
    flag_binarize_data = 1;
elseif(model_id==2)
    flag_normalize_data = 1;
elseif(model_id==3)
    % no change
elseif(model_id==4)
    flag_binarize_data = 1;
else
    error(['Unknown model id: ' num2str(model_id)]);
end

if(flag_binarize_data)
    if(data_id==1)
        data.X = load('dat/mnist/binarized_mnist_train.amat');
        [data.N, data.D] = size(data.X);
        data.test.X = load('dat/mnist/binarized_mnist_test.amat');
        data.test.N = size(data.test.X, 1);
    elseif(data_id==2)
        data.X = loadMNISTImages('dat/fashionmnist/train-images-idx3-ubyte')';
        data.X = double(data.X>0.5);
        [data.N, data.D] = size(data.X);        % Binarize
        data.test.X = loadMNISTImages('dat/fashionmnist/t10k-images-idx3-ubyte')';
        data.test.X = double(data.test.X>0.5);  % Binarize
        data.test.N = size(data.test.X, 1);
    else
        error(['Unknown data id: ' num2str(data_id)]);
    end
else
    if(data_id==1)
        load dat/mnist/mnist_all.mat
        
        data.X = double([train0; train1; train2; train3; train4; train5; train6; train7; train8; train9]);
        [data.N, data.D] = size(data.X);
        data.test.X = double([test0; test1; test2; test3; test4; test5; test6; test7; test8; test9]);
        data.test.N = size(data.test.X, 1);
        clear train*
        clear test*
    elseif(data_id==2)
        data.X = loadMNISTImages('dat/fashionmnist/train-images-idx3-ubyte')';
        data.X = 255*data.X;
        [data.N, data.D] = size(data.X);        % Counts from 0-255
        data.test.X = loadMNISTImages('dat/fashionmnist/t10k-images-idx3-ubyte')';
        data.test.X = 255*data.test.X;          % Counts from 0-255
        data.test.N = size(data.test.X, 1);
    else
        error(['Unknown data id: ' num2str(data_id)]);
    end
    
    if(flag_normalize_data)
        % Real-valued data in the interval [0,1]
        data.X = data.X/255;
        meanX = mean(data.X, 1);
        data.X = bsxfun(@minus, data.X, meanX);
        data.test.X = data.test.X/255;
        data.test.X = bsxfun(@minus, data.test.X, meanX);
    end
end

% Input for the recognition networks
if(flag_binarize_data || flag_normalize_data)
    data.Xinput = data.X;
    data.test.Xinput = data.test.X;
else
    % Normalize the inputs to be in [0,1]
    if(data_id==1 || data_id==2)
        data.Xinput = data.X/255;
        data.test.Xinput = data.test.X/255;
    else
        error(['Unknown data id: ' num2str(data_id)]);
    end
end

% Pre-compute log(X!) for poissMF
if(model_id==3)
    data.logfactX = gammaln(data.X+1);
    data.test.logfactX = gammaln(data.test.X+1);
end

%% Joint distribution
if(model_id==1)
    % Variational autoencoder 
    % Create the decoder neural net (i.e., the model pxz)
    numUnitsPerHiddenLayer = [200 200];
    inputDim = dim_z;       % latent dimensionality 
    outputDim = data.D;     % output/data dimensionality 
    act_funcs = {'sigmoid', 'relu', 'relu'};  % the output activation is the simgoid since the data is binary
    numUnitsPerLayer = [outputDim numUnitsPerHiddenLayer inputDim];  % all units from output (left) to the input (right)
    vae = netcreate(numUnitsPerLayer, act_funcs); 
    vae{1}.lik = 'Bernoulli';   % since data is binary
elseif(model_id==2)
    % Gaussian Matrix Factorization
    numUnitsPerHiddenLayer = [];
    inputDim = dim_z;       % number of latent factors
    outputDim = data.D;     % output/data dimensionality 
    act_funcs = {'lin'};    % linear transformations only
    numUnitsPerLayer = [outputDim numUnitsPerHiddenLayer inputDim];  % all units from output (left) to the input (right)
    vae = netcreate(numUnitsPerLayer, act_funcs); 
    vae{1}.lik = 'Gaussian';
    vae{1}.logbeta = -log(variance_lik);
    vae{1}.b(:) = 0;        % set all the intercepts to 0
elseif(model_id==3)
    % Poisson Matrix Factorization
    numUnitsPerHiddenLayer = [];
    inputDim = dim_z;       % number of latent factors
    outputDim = data.D;     % output/data dimensionality 
    act_funcs = {'softplus_threshold'};
    numUnitsPerLayer = [outputDim numUnitsPerHiddenLayer inputDim];  % all units from output (left) to the input (right)
    vae = netcreate(numUnitsPerLayer, act_funcs); 
    vae{1}.lik = 'Poisson';
elseif(model_id==4)
    % Logistic Matrix Factorization
    numUnitsPerHiddenLayer = [];
    inputDim = dim_z;       % number of latent factors
    outputDim = data.D;     % output/data dimensionality 
    act_funcs = {'sigmoid'};
    numUnitsPerLayer = [outputDim numUnitsPerHiddenLayer inputDim];  % all units from output (left) to the input (right)
    vae = netcreate(numUnitsPerLayer, act_funcs); 
    vae{1}.lik = 'Bernoulli';
else
    error(['Unknown model id: ' num2str(model_id)]);
end
% Log-joint density p(x,z) 
pxz.logdensity = @logdensityVAE;  
pxz.inargs{1} = vae; 

%% Variational distribution - Amortized Gaussian

% Set the number of hidden units
numUnitsPerHiddenLayer = [200 200];  % units per hidden layer (the length of this vector is the number of hidden layers)

% Create the neural net for the mean
inputDim = data.D;
outputDim = dim_z;
numUnitsPerLayer = [outputDim numUnitsPerHiddenLayer inputDim];  % all units from output (left) to the input (right)
act_funcs = {'lin', 'relu', 'relu'}; % possible choices for the activations functions: 'relu' 'softmax' 'lin' 'cos' 'sigmoid' 'tanh' 'softplus' 'softplus_threshold'
netMu = netcreate(numUnitsPerLayer, act_funcs);

% Create the neural net for the variance
inputDim = data.D;
outputDim = dim_z;
numUnitsPerLayer = [outputDim numUnitsPerHiddenLayer inputDim];  % all units from output (left) to the input (right)
act_funcs = {'softplus_threshold', 'relu', 'relu'}; % positive output since this is going to represent std values
netSigma = netcreate(numUnitsPerLayer, act_funcs);

% Define the NN
vardist.netMu = netMu;
vardist.netSigma = netSigma;
clear netMu netSigma;

%% MCMC algorithm
AdaptDuringBurn = 1;
LF = 5;  % leap frog steps
mcmc.algorithm = @hmc_vae;   % other options: @metropolisHastings; @mala;
mcmc.inargs{1} = 0.5/dim_z;  % initial step size parameter delta
mcmc.inargs{2} = BurnIters;
mcmc.inargs{3} = SamplingIters;
mcmc.inargs{4} = AdaptDuringBurn;
mcmc.inargs{5} = LF;

%% RMSprop variables

% For the variational distribution (encoder)
Gt.netMu = cell(1, length(vardist.netMu)-1);
for layer=1:length(vardist.netMu)-1
   Gt.netMu{layer}.W = 0.01*ones(size( vardist.netMu{layer}.W ));
   Gt.netMu{layer}.b = 0.01*ones(size( vardist.netMu{layer}.b ));
end
Gt.netSigma = cell(1, length(vardist.netSigma)-1);
for layer=1:length(vardist.netSigma)-1
   Gt.netSigma{layer}.W = 0.01*ones(size( vardist.netSigma{layer}.W ));
   Gt.netSigma{layer}.b = 0.01*ones(size( vardist.netSigma{layer}.b ));
end

% For the model parameter (decoder)
Gt.vae = cell(1, length(vae)-1);
for layer=1:length(vae)-1
    Gt.vae{layer}.W = 0.01*ones(size( vae{layer}.W ));
    Gt.vae{layer}.b = 0.01*ones(size( vae{layer}.b ));
end

%% Main algorithm

% Number of iterations
iters = 400000;

% Values to compute at each iteration
stochasticDiv = zeros(1, iters);
ELBO_q = zeros(1, iters);
expLogLik_qt = zeros(1, iters);
first_term = zeros(1, iters);
telapsed = zeros(1, iters);
test_loglik = zeros(1, iters);

% Average acceptance history and rate for all MCMC chains
acceptHist = zeros(data.N, BurnIters+SamplingIters);
acceptRate = zeros(data.N, 1);

% Initialize control variates
control_variate = zeros(data.N, 1);

% Algorithm
for it=1:iters
%   
    t_start = tic;
    
    % Take the next minibatch 
    if(it==1)
        block = takeNextBatch(data.N, batchsize, 1, randperm(data.N));
    else
        block = takeNextBatch(data.N, batchsize);
    end
    pxz.inargs{1} = vae;
    pxz.inargs{1}{1}.outData = data.X(block,:); % place the minibatch in the model 
    if(model_id==3)
        pxz.inargs{1}{1}.logfactX = data.logfactX(block,:); % place the log(X!) terms of the minibatch
    end

    % Sample z~q(z|x)
    netMu = netforward(vardist.netMu, data.Xinput(block,:));
    netSigma = netforward(vardist.netSigma, data.Xinput(block,:));
    eta = randn(batchsize, dim_z);
    z = netMu{1}.Z + bsxfun(@times, eta, netSigma{1}.Z);
    
    % Methods 1 & 2: VCD(+controlVar)
    if(method_id==1 || method_id==2)
        % First term: Expectation over q(z)
        % Evaluate the log-density and the stochastic gradients
        [logpxz, gradz] = pxz.logdensity(z, pxz.inargs{:});
        precond_mu_first = gradz;                             % grad assuming analytic entropy
        precond_sigma_first = gradz.*eta;                     % grad assuming analytic entropy
        first_term(it) = mean(logpxz) + 0.5*dim_z;            % all the other terms from the entropy are cancelled out since
                                                              % they appear with opposite sigh in the second term
        
        % Improve the samples z by drawing zt~Q(zt|z) using HMC
        [zt, ~, extraOutputs] = mcmc.algorithm(z, pxz, mcmc.inargs{:});
        % Keep track of acceptance rate (for information purposes only)
        acceptHist(block, :) = acceptHist(block, :) + (data.N/batchsize)*extraOutputs.acceptHist/iters;
        acceptRate(block) = acceptRate(block) + (data.N/batchsize)*extraOutputs.accRate/iters;
        % Adapt the stepsize for HMC
        mcmc.inargs{1} = extraOutputs.delta;

        % Compute the second and third terms of the gradient
        [logpx_zt, ~, gradW_modelParams, gradb_modelParams] = pxz.logdensity(zt, pxz.inargs{:});
        diff = (zt - netMu{1}.Z)./netSigma{1}.Z;
        diff2 = diff.^2;
        f_zt = logpx_zt + 0.5*sum(diff2, 2);
        precond_mu_second = bsxfun(@times, eta./netSigma{1}.Z, f_zt-control_variate(block));
        precond_sigma_second = bsxfun(@times, -1./netSigma{1}.Z + (eta.^2)./netSigma{1}.Z, f_zt-control_variate(block));
        precond_mu_third = diff./netSigma{1}.Z;
        precond_sigma_third = diff2./netSigma{1}.Z;

        % Backpropagate through the network to obtain the gradients wrt the network parameters (W and b)
        [grad_mu_W, grad_mu_b] = netbackpropagation(netMu, precond_mu_first - precond_mu_second + precond_mu_third, 1);
        [grad_sigma_W, grad_sigma_b] = netbackpropagation(netSigma, precond_sigma_first - precond_sigma_second + precond_sigma_third, 0);

        % Stochastic divergence
        stochasticDiv(it) = -first_term(it) + mean(f_zt);
    
    % Method 3: Standard KL
    elseif(method_id==3)
        % Evaluate the stochastic gradients
        [logpxz, gradz, gradW_modelParams, gradb_modelParams] = pxz.logdensity(z, pxz.inargs{:}); 
        precond_grad = gradz;
        
        % Backpropagate through the network to obtain the gradients wrt the network parameters (W and b)
        [grad_mu_W, grad_mu_b] = netbackpropagation(netMu, precond_grad, 1);
        [grad_sigma_W, grad_sigma_b] = netbackpropagation(netSigma, precond_grad.*eta + 1./netSigma{1}.Z, 0);
        
        logpx_zt = logpxz;
        
    % Method 4: Hoffman's MCMC method
    elseif(method_id==4)
        % Evaluate the log-density and the stochastic gradients
        [logpxz, gradz] = pxz.logdensity(z, pxz.inargs{:});
        precond_mu_first = gradz;                             % grad assuming analytic entropy
        precond_sigma_first = gradz.*eta + 1./netSigma{1}.Z;  % grad assuming analytic entropy
        
        % Improve the samples z by drawing zt~Q(zt|z) using HMC
        [zt, ~, extraOutputs] = mcmc.algorithm(z, pxz, mcmc.inargs{:});
        % Keep track of acceptance rate (for information purposes only)
        acceptHist(block, :) = acceptHist(block, :) + (data.N/batchsize)*extraOutputs.acceptHist/iters;
        acceptRate(block) = acceptRate(block) + (data.N/batchsize)*extraOutputs.accRate/iters;
        % Adapt the stepsize for HMC
        mcmc.inargs{1} = extraOutputs.delta;
        
        % Compute the gradient wrt the model parameters
        [logpx_zt, ~, gradW_modelParams, gradb_modelParams] = pxz.logdensity(zt, pxz.inargs{:});

        % Backpropagate through the network to obtain the gradients wrt the network parameters (W and b)
        % (there is no feedback from the improved sample zt in this step)
        [grad_mu_W, grad_mu_b] = netbackpropagation(netMu, precond_mu_first, 1);
        [grad_sigma_W, grad_sigma_b] = netbackpropagation(netSigma, precond_sigma_first, 0);
        
    else
        error(['Unknown method id: ' num2str(method_id)]);
    end

    % RMSprop update of the variational parameters
    kappa = kappa0;
    if(it==1)
        kappa = 1;
    end
    for layer=length(netMu)-1:-1:1
        Gt.netMu{layer}.W = kappa*(grad_mu_W{layer}.^2) + (1-kappa)*Gt.netMu{layer}.W;
        Gt.netMu{layer}.b = kappa*(grad_mu_b{layer}.^2) + (1-kappa)*Gt.netMu{layer}.b;
        Gt.netSigma{layer}.W = kappa*(grad_sigma_W{layer}.^2) + (1-kappa)*Gt.netSigma{layer}.W;
        Gt.netSigma{layer}.b = kappa*(grad_sigma_b{layer}.^2) + (1-kappa)*Gt.netSigma{layer}.b;

        vardist.netMu{layer}.W = vardist.netMu{layer}.W + rhotheta*grad_mu_W{layer}./(TT+sqrt(  Gt.netMu{layer}.W ));
        vardist.netMu{layer}.b = vardist.netMu{layer}.b + rhotheta*grad_mu_b{layer}./(TT+sqrt(  Gt.netMu{layer}.b ));
        vardist.netSigma{layer}.W = vardist.netSigma{layer}.W + rhosigma*grad_sigma_W{layer}./(TT+sqrt(  Gt.netSigma{layer}.W ));
        vardist.netSigma{layer}.b = vardist.netSigma{layer}.b + rhosigma*grad_sigma_b{layer}./(TT+sqrt(  Gt.netSigma{layer}.b ));
    end

    % RMSprop update of the model/decoder parameters
    for layer=length(vae)-1:-1:1
        Gt.vae{layer}.W = kappa*(gradW_modelParams{layer}.^2) + (1-kappa)*Gt.vae{layer}.W;
        Gt.vae{layer}.b = kappa*(gradb_modelParams{layer}.^2) + (1-kappa)*Gt.vae{layer}.b;
        vae{layer}.W = vae{layer}.W + rhoModelParams*gradW_modelParams{layer}./(TT+sqrt(  Gt.vae{layer}.W ));
        vae{layer}.b = vae{layer}.b + rhoModelParams*gradb_modelParams{layer}./(TT+sqrt(  Gt.vae{layer}.b ));
        if(model_id==2)
            % For gaussMF, keep the intercepts at 0
            vae{layer}.b(:) = 0;
        end
    end
    
    % Estimate entropy and ELBO
    entropy = 0.5*dim_z*log(2*pi) + sum(log(netSigma{1}.Z),2) + dim_z/2;
    ELBO_q(it) = mean(logpxz) + mean(entropy);
    expLogLik_qt(it) = mean(logpx_zt);
    if(mod(it,100)==0)
        fprintf('Iter=%d, Stoch VCD=%f, ELBO_q=%f, Stoch LogLik_qt=%f', it, stochasticDiv(it), ELBO_q(it), expLogLik_qt(it));
    end
    
    % Update the local control variates
    if(flag_control_variate)
        if(it>control_iters_wait)
            control_variate(block) = decay_control_variate*control_variate(block) + (1-decay_control_variate)*f_zt;
        else
            control_variate(:) = decay_control_variate*control_variate(:) + (1-decay_control_variate)*mean(f_zt);
        end
    end
    
    % Reduce stepsize every 'ReducedEvery' iterations
    if mod(it, ReducedEvery) == 0
        rhosigma = rhosigma*ReducedBy;
        rhotheta = rhotheta*ReducedBy;
        rhoModelParams = rhoModelParams*ReducedBy;
    end
    
    % Elapsed time
    telapsed(it) = toc(t_start);
    
    % Evaluate test log-likelihood
    flag_compute_llh = 0;
    if(it==iters)
        flag_compute_llh = 1;
        S = 20000;
    elseif(mod(it, 100000)==0)
        flag_compute_llh = 1;
        S = 5;
    end
    if(flag_compute_llh)
        pxz.vae = vae;
        if(model_id==2)
            % For Gaussian MF, compute the marginal likelihood analytically
            test_loglik_0 = compute_llh_gaussMF(pxz, data);
            test_loglik_1 = test_loglik_0;
            test_loglik_2 = test_loglik_0;
        else
            % For other models, use three different proposals
            [test_loglik_0 test_loglik_0_all] = compute_llh_vae_explicit(S, pxz, vardist, data);
            [test_loglik_1 test_loglik_1_all] = compute_llh_vae_explicit_useHMC(S, pxz, vardist, data, mcmc, 300, 300, 0);
            [test_loglik_2 test_loglik_2_all] = compute_llh_vae_explicit_useHMC(S, pxz, vardist, data, mcmc, 300, 300, 1);
        end
        test_loglik(it) = test_loglik_2;
        fprintf(', test log-lik=%f', test_loglik(it));
    end
    if(mod(it,100)==0)
        fprintf('\n');
    end
%   
end

%% Save results
save([outdir outName '_results.mat'], '-v7.3');

%% Plots

FontSz = 18;

% Plot the smoothed VCD
smoothed_stochasticDiv = smoothedAverage(stochasticDiv, 200);
figure;
plot(smoothed_stochasticDiv, 'r', 'linewidth', 2);
axis([1 iters 0 2000])
set(gca,'fontsize',FontSz);
box on;
name = [outdir outName '_newDivValue'];
print('-depsc2', '-r300', name);
cmd = sprintf('epstopdf %s', [name '.eps']);
system(cmd);

if(data_id==1 || data_id==2)
    % Reconstructed data by sampling from the improved variational distribution qt(z)
    rand_idx = randperm(data.test.N);
    data.test.X = data.test.X(rand_idx,:);
    data.test.Xinput = data.test.Xinput(rand_idx,:);
    if(model_id==3)
        data.test.logfactX = data.test.logfactX(rand_idx,:);
    end
    netMu = netforward(vardist.netMu, data.Xinput(1:200,:));
    netSigma = netforward(vardist.netSigma, data.Xinput(1:200,:));
    eta = randn(200,dim_z);
    z = netMu{1}.Z + bsxfun(@times, eta, netSigma{1}.Z);
    if(method_id==1 || method_id==2 || method_id==4)
        pxz.inargs{1} = vae;
        pxz.inargs{1}{1}.outData = data.X(1:200,:);
        if(model_id==3)
            pxz.inargs{1}{1}.logfactX = data.logfactX(1:200,:);
        end
        [zt, ~, ~] = mcmc.algorithm(z, pxz, mcmc.inargs{:});
        netReco = netforward(vae, zt);
    elseif(method_id==3)
        netReco = netforward(vae, z);
    else
        error(['Unknown method id: ' num2str(method_id)]);
    end

    % Plot reconstructed data
    S = 10; 
    figure;
    cnt = 0; 
    for i=1:S  
        cnt = cnt + 1;
        subtightplot(S,S,cnt);     
        imagesc(reshape(data.X(i,:),sqrt(data.D),sqrt(data.D))');
        axis off;
        colormap('gray'); 
        subtightplot(S,S,S+cnt);     
        imagesc(reshape(netReco{1}.Z(i,:), sqrt(data.D),sqrt(data.D))');
        axis off;
        colormap('gray'); 
    end
    set(gca,'fontsize',FontSz);
    box on;
    name = [outdir outName 'Reconstruct'];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);
else
    error(['Unknown data id: ' num2str(data_id)]);
end
