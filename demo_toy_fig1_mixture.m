clear;
close all;

% Set the seed
randn('seed', 1);
rand('seed', 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% CHOOSE ONE %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataName = 'banana2D';
%dataName = 'mixture2Dgauss';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath nnet; 
addpath mcmc;
addpath aux;
addpath logdensities;
outdir= 'plots/';
if(~isdir(outdir))
    mkdir(outdir);
end

% Color definitions
my_green = [0.466 0.674 0.188];   % green
my_blue = [0 0.447 0.741];        % blue
my_orange = [0.929 0.694 0.125];  % orange
my_red = [0.85 0.325 0.098];      % red
my_purple = [0.494 0.184 0.556];  % purple

FontSz = 18;
print_legend = 1;

%% Model joint distribution

dim_z = 2; % dimensionality of z

if(strcmp(dataName, 'mixture2Dgauss'))
    weights = [0.3 0.7];
    mu = [0.8 0.8; -2 -2];
    Sigma = cat(3, [1 0.8; 0.8 1], [1 -0.6; -0.6 1]);
    L = zeros(size(Sigma));
    for cc=1:size(Sigma,3)
        L(:,:,cc) = chol(Sigma(:,:,cc))';
    end
    pxz.logdensity = @logdensityGaussianMixture;
    pxz.inargs{1} = weights;  % mixture weights
    pxz.inargs{2} = mu;       % mean vectors
    pxz.inargs{3} = L;        % Cholesky decomposition of the covariance matrices
    
    xMin = -4.2; xMax = 3.2;     % For plotting
    yMin = -4.2; yMax = 3.2;
elseif(strcmp(dataName, 'banana2D'))
    muBanana = [0 0];
    Sigma = [1 0.9; 0.9 1];
    L = chol(Sigma)';
    a = 1;
    b = 1;
    pxz.logdensity = @logdensityBanana2D; 
    pxz.inargs{1} = muBanana;  % mean vector data 
    pxz.inargs{2} = L;   % Cholesky decomposition of the covariacne matrix 
    pxz.inargs{3} = a;   % 1st bananity parameter  
    pxz.inargs{4} = b;   % 2nd bananity parameter
    
    xMin = -2; xMax = 2;      % For plotting
    yMin = -6.5; yMax = 1;
end

%% Variational distribution - Mixture of Gaussians

vardist.components = 2;
vardist.weights = (1/vardist.components)*ones(1,vardist.components);
softmax_weights = zeros(1,vardist.components);
vardist.mu = randn(vardist.components, dim_z);
vardist.sigma = 0.1*ones(vardist.components, dim_z);

vardistInit = vardist;

% MCMC algorithm
BurnIters = 2;       % Burn-in samples
SamplingIters = 1;   % Samples
AdaptDuringBurn = 1;
LF = 5;  % leap frog steps
mcmc.algorithm = @hmc; % @metropolisHastings; @mala;
mcmc.inargs{1} = 0.5/dim_z; % initial step size parameter delta
mcmc.inargs{2} = BurnIters;
mcmc.inargs{3} = SamplingIters;
mcmc.inargs{4} = AdaptDuringBurn;
mcmc.inargs{5} = LF;

%% Algorithm parameters

iters = 50000;   % Number of iterations

% RMSProp parameters
rhomu = 0.01;
rhosigma = 0.005;
rhoweights = 0.001;
ReducedBy = 0.9; 
ReducedEvery = 2000;
TT = 1;
kappa0 = 0.1;

% Control variates
control_variate = 0;
theta = 0.9;
use_control_variate = 1;

% Quantities to compute at each iteration
stochasticDiv = zeros(1, iters);
first_term = zeros(1, iters);
second_third_terms = zeros(1, iters);

% Acceptance rate of HMC
acceptHist = zeros(1, BurnIters+SamplingIters);
acceptRate = 0;

% Variables for RMSProp
Gt_w = 0.01*ones(size( vardist.weights ));
Gt_mu = 0.01*ones(size( vardist.mu ));
Gt_sigma = 0.01*ones(size( vardist.sigma ));
Gt_weights = 0.01*ones(size( vardist.weights ));

%% Algorithm
tic;
for it=1:iters 
%
    
    % Auxiliary variables to store the gradients
    precond_mu_first = zeros(vardist.components, dim_z);
    precond_sigma_first = zeros(vardist.components, dim_z);
    precond_weights_first = zeros(1, vardist.components);
    precond_mu_second = zeros(vardist.components, dim_z);
    precond_sigma_second = zeros(vardist.components, dim_z);
    precond_weights_second = zeros(1, vardist.components);
    precond_mu_third = zeros(vardist.components, dim_z);
    precond_sigma_third = zeros(vardist.components, dim_z);
    precond_weights_third = zeros(1, vardist.components);
    
    dlogq0_dmu = zeros(vardist.components, dim_z);
    dlogq0_dsigma = zeros(vardist.components, dim_z);
    fzt_c = zeros(1, vardist.components);
    for cc=1:vardist.components
        
        % Sample z
        eta = randn(1,dim_z);
        z = vardist.mu(cc,:) + vardist.sigma(cc,:).*eta;
        % Evaluate log-densities logp(x,z) and logq(z)
        [logp, dlogp_dz] = pxz.logdensity(z, pxz.inargs{:});
        [logq, dlogq_dz] = logdensityGaussianDiagonalMixture(z, vardist.weights, vardist.mu, vardist.sigma);
        % Obtain gradients and stochastic estimate of the term
        precond_mu_first(cc,:) = vardist.weights(cc) * ( dlogp_dz - dlogq_dz );
        precond_sigma_first(cc,:) = vardist.weights(cc) * ( (dlogp_dz - dlogq_dz).*eta );
        precond_weights_first(cc) = ( logp - logq );
        
        first_term(it) = first_term(it) + vardist.weights(cc) * ( logp - logq );
        
        % Improve the sample by drawing zt~Q(zt|z) using HMC
        [zt, samples, extraOutputs] = mcmc.algorithm(z, pxz, mcmc.inargs{:});
        % Keep track of acceptance rate (for information purposes only)
        acceptHist = acceptHist + extraOutputs.acceptHist/iters;
        acceptRate = acceptRate + extraOutputs.accRate/iters;
        % Adapt the stepsize for HMC
        mcmc.inargs{1} = extraOutputs.delta;
        
        % Evaluate the densities and the gradient (second expectation)
        logp_zt = pxz.logdensity(zt, pxz.inargs{:});
        logq_zt = logdensityGaussianDiagonalMixture(zt, vardist.weights, vardist.mu, vardist.sigma);
        fzt = logp_zt - logq_zt;
        fzt_c(cc) = fzt;
        weight_fzt = vardist.weights(cc) * (fzt - control_variate);
        % nabla_theta logq_c(z_0)
        dlogq0_dmu(cc,:) = eta./vardist.sigma(cc,:);
        dlogq0_dsigma(cc,:) = (eta.^2)./vardist.sigma(cc,:) - 1./vardist.sigma(cc,:);
        precond_mu_second(cc, :) =  weight_fzt * dlogq0_dmu(cc,:);
        precond_sigma_second(cc,:) = weight_fzt * dlogq0_dsigma(cc,:);
        precond_weights_second(cc) = fzt;
        
        % Evaluate the gradient (third expectation)
        % nabla_theta logq(z_t)
        aux = bsxfun(@minus, zt, vardist.mu)./vardist.sigma;
        aux2 = aux.^2;
        loggauss_c = log(vardist.weights)' - sum(log(vardist.sigma),2) - 0.5*sum(aux2,2);
        p_comp = softmax(loggauss_c, 1);
        dlogq_dmu = bsxfun(@times, p_comp, aux./vardist.sigma);
        dlogq_dsigma = bsxfun(@times, p_comp, aux2./vardist.sigma - 1./vardist.sigma);
        precond_mu_third = precond_mu_third + vardist.weights(cc)*dlogq_dmu;
        precond_sigma_third = precond_sigma_third + vardist.weights(cc)*dlogq_dsigma;
        precond_weights_third = precond_weights_third + vardist.weights(cc) * p_comp';
             
        % Objective
        second_third_terms(it) = second_third_terms(it) + vardist.weights(cc) * fzt;
    end

    % Renormalize weights w.r.t. the weights
    precond_weights_third = precond_weights_third./vardist.weights;
    
    % Update the control variate
    if use_control_variate == 1
        control_variate = theta*control_variate + (1-theta)*sum(vardist.weights.*fzt_c);
    end
    
    % Total gradient
    grad_mu = precond_mu_first - precond_mu_second + precond_mu_third;
    grad_sigma = precond_sigma_first - precond_sigma_second + precond_sigma_third;
    grad_weights = precond_weights_first - precond_weights_second + precond_weights_third;
            
    % RMSprop update of the variational parameters
    kappa = kappa0;
    if(it==1)
        kappa = 1;
    end
    Gt_mu = kappa*(grad_mu.^2) + (1-kappa)*Gt_mu;
    vardist.mu = vardist.mu + rhomu*grad_mu./(TT+sqrt(  Gt_mu ));
    Gt_sigma = kappa*(grad_sigma.^2) + (1-kappa)*Gt_sigma;
    vardist.sigma = vardist.sigma + rhosigma*grad_sigma./(TT+sqrt(  Gt_sigma ));
    vardist.sigma(vardist.sigma<0.00001) = 0.00001; % for numerical stability
    % Convert the gradient of the weights wrt gradient wrt the unnormalized weights
    grad_weights = ((-vardist.weights'*vardist.weights + diag(vardist.weights)) * grad_weights')';
    Gt_weights = kappa*(grad_weights.^2) + (1-kappa)*Gt_weights;
    softmax_weights = softmax_weights + rhoweights*grad_weights./(TT+sqrt( Gt_weights ));
    vardist.weights = softmax(softmax_weights, 2);
            
    % Stochastic divergence
    stochasticDiv(it) = - first_term(it) + second_third_terms(it);
    if mod(it,1000) == 0
        fprintf('Iter=%d, Stoch VCD=%f\n', it, stochasticDiv(it));
    end
    
    % Decrease learning rates
    if mod(it, ReducedEvery) == 0
        rhosigma = rhosigma*ReducedBy;
        rhomu = rhomu*ReducedBy;
        rhoweights = rhoweights*ReducedBy;
    end
%   
end
timetaken = toc; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Also fit the target with standard VI

vardistVCD = vardist;
vardist = vardistInit;

% Algorithm parameters
iters = 50000;
ELBO_q = zeros(1, iters);
rhomu = 0.01;
rhosigma = 0.005;
ReducedBy = 0.9;
ReducedEvery = 2000;
TT = 1;
kappa0 = 0.1;
Gt_mu = 0.01*ones(size( vardist.mu ));
Gt_sigma = 0.01*ones(size( vardist.sigma ));
Gt_weights = 0.01*ones(size( vardist.weights ));

tic;
for it=1:iters
%
    grad_mu = zeros(vardist.components, dim_z);
    grad_sigma = zeros(vardist.components, dim_z);
    grad_weights = zeros(1, vardist.components);
    for cc=1:vardist.components
        % Sample z
        eta = randn(1,dim_z);
        z = vardist.mu(cc,:) + vardist.sigma(cc,:).*eta;
        [logp, dlogp_dz] = pxz.logdensity(z, pxz.inargs{:});
        [logq, dlogq_dz] = logdensityGaussianDiagonalMixture(z, vardist.weights, vardist.mu, vardist.sigma);
        grad_mu(cc,:) = vardist.weights(cc) * ( dlogp_dz - dlogq_dz );
        grad_sigma(cc,:) = vardist.weights(cc) * ( (dlogp_dz - dlogq_dz).*eta );
        grad_weights(cc) = ( logp - logq );
        
        ELBO_q(it) = ELBO_q(it) + vardist.weights(cc) * (logp - logq);
    end

    % RMSprop update of the variational parameters
    kappa = kappa0;
    if(it==1)
      kappa = 1;
    end
    Gt_mu = kappa*(grad_mu.^2) + (1-kappa)*Gt_mu;
    vardist.mu = vardist.mu + rhomu*grad_mu./(TT+sqrt(  Gt_mu ));

    Gt_sigma = kappa*(grad_sigma.^2) + (1-kappa)*Gt_sigma;
    vardist.sigma = vardist.sigma + rhosigma*grad_sigma./(TT+sqrt(  Gt_sigma ));
    vardist.sigma(vardist.sigma<0.00001) = 0.00001; % for numerical stability
    
    % Only update the weights after some initial iterations
    if(it > 1000)
        % Convert the gradient of the weights wrt gradient wrt the unnormalized weights
        grad_weights = ((-vardist.weights'*vardist.weights + diag(vardist.weights)) * grad_weights')';
        Gt_weights = kappa*(grad_weights.^2) + (1-kappa)*Gt_weights;
        aux_new_weights = log(vardist.weights) + rhoweights*grad_weights./(TT+sqrt( Gt_weights )); 
        vardist.weights = softmax(aux_new_weights, 2);
    end

    if mod(it,1000) == 0
        fprintf('Iter=%d, ELBO_q=%f\n', it, ELBO_q(it));
    end

    if mod(it, ReducedEvery) == 0
        rhosigma = rhosigma*ReducedBy;
        rhomu = rhomu*ReducedBy;
        rhoweights = rhoweights*ReducedBy;
    end
%
end
timetaken = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plots

% Plot the smoothed VCD
smoothed_stochasticDiv = smoothedAverage(stochasticDiv, 200);
figure;
plot(smoothed_stochasticDiv,'r', 'linewidth',2);
set(gca,'fontsize',FontSz);
box on;
axis([1 iters 0 5]);
name = [outdir dataName '_NumComps_' num2str(vardist.components) '_VCD'];
print('-depsc2', '-r300', name);
cmd = sprintf('epstopdf %s', [name '.eps']);
system(cmd);

% Plot the smoothed ELBO for the standard VI
smoothed_elbo_q = smoothedAverage(ELBO_q, 200);
figure;
plot(smoothed_elbo_q, 'b', 'linewidth',2);
set(gca,'fontsize',FontSz);
box on;
name = [outdir dataName '_NumComps_' num2str(vardist.components) '_standardVI_elbo'];
print('-depsc2', '-r300', name);
cmd = sprintf('epstopdf %s', [name '.eps']);
system(cmd);

% Plot the densities
figure;
hold on;
x = linspace(xMin, xMax, 300);
y = linspace(yMin, yMax, 300);
[X,Y] = meshgrid(x,y);
Z = zeros(length(x), length(y));
ZvarSimple = zeros(length(x), length(y));
ZvarVCD = zeros(length(x), length(y));
for i=1:length(x)
    for j=1:length(y)
        Z(i,j) = pxz.logdensity([x(i), y(j)], pxz.inargs{:});
        ZvarSimple(i,j) = logdensityGaussianDiagonalMixture([x(i), y(j)], vardist.weights, vardist.mu, vardist.sigma );
        ZvarVCD(i,j) = logdensityGaussianDiagonalMixture([x(i), y(j)], vardistVCD.weights, vardistVCD.mu, vardistVCD.sigma );
    end
end
[~, h_target] = contour(X, Y, exp(Z)', 4, 'Color', my_green, 'Linewidth', 1.2);
[~, h_standard] = contour(X,Y,exp(ZvarSimple)', 3, 'Color', my_blue, 'Linewidth', 1.2);
[~, h_newdiv] = contour(X,Y,exp(ZvarVCD)', 3, 'Color', my_red, 'Linewidth', 1.2);
if(print_legend)
    legend_str = {'Target', 'Standard KL', 'VCD'};
    legend([h_target, h_standard, h_newdiv], legend_str, 'Location', 'SouthEast', 'box', 'off', 'EdgeColor', 'white');
    legend boxoff;
end
name = [outdir dataName '_NumComps_' num2str(vardist.components) '_VCDvsStandardVI'];
print('-depsc2', '-r300', name);
cmd = sprintf('epstopdf %s', [name '.eps']);
system(cmd);
% Also export as pdf
set(gca, 'box', 'off');
figurepdf(3, 2.33);
print('-dpdf', [name '.pdf']);
