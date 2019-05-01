clear;
close all;

% Set the seed
randn('seed',2);
rand('seed',2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% CHOOSE ONE %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%dataName = 'twoDgaussian';
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

%% Model joint distribution

dim_z = 2; % dimensionality of z

if(strcmp(dataName, 'twoDgaussian'))
    mu = [0 0];
    Sigma = [1 0.95; 0.95 1];
    L = chol(Sigma)';
    pxz.logdensity = @logdensityGaussian;
    pxz.inargs{1} = mu;  % mean vector data
    pxz.inargs{2} = L;   % Cholesky decomposition of the covariacne matrix
    
    xMin = -2; xMax = 2; % For plotting
    yMin = -2; yMax = 2;
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
elseif (strcmp(dataName, 'mixture2Dgauss'))
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
    xMin = -4.2; xMax = 3.2;      % For plotting
    yMin = -4.2; yMax = 3.2;
end

%% Variational distribution - Factorized Gaussian

vardist.mu = randn(1,dim_z);
vardist.sigma = 0.1*ones(1,dim_z);

vardistInit = vardist;

% MCMC algorithm
BurnIters = 2;       % Burn-in samples
SamplingIters = 1;   % Samples
AdaptDuringBurn = 1;
LF = 5;  % leap frog steps
mcmc.algorithm = @hmc; %@metropolisHastings; @mala;
mcmc.inargs{1} = 0.5/dim_z; % initial step size parameter delta
mcmc.inargs{2} = BurnIters;
mcmc.inargs{3} = SamplingIters;
mcmc.inargs{4} = AdaptDuringBurn;
mcmc.inargs{5} = LF;

%% Algorithm parameters

iters = 20000;  % Number of iterations

% RMSProp parameters
rhomu = 0.01;
rhosigma = 0.005;
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
ELBO_q = zeros(1, iters);
expLogLik_qt = zeros(1, iters);
first_term = zeros(1, iters);

% Acceptance rate of HMC
acceptHist = zeros(1, BurnIters+SamplingIters);
acceptRate = 0;

% Variables for RMSProp
Gt_mu = 0.01*ones(size( vardist.mu ));
Gt_sigma = 0.01*ones(size( vardist.sigma ));

%% Algorithm
tic;
for it=1:iters 
%
    % Sample z
    eta = randn(1,dim_z);
    z = vardist.mu + vardist.sigma.*eta;
    % Evaluate log-densities logp(x,z) and (constant+)logq(z)
    [logpxz, gradz] = pxz.logdensity(z, pxz.inargs{:});
    % Obtain gradients and stochastic estimate of the term
    precond_mu_first = gradz;                           % grad assuming analytic entropic term
    precond_sigma_first = gradz.*eta;                   % grad assuming analytic entropy term (after cancelling the constant wrt z terms)
    first_term(it) = logpxz + 0.5*dim_z;                % (i.e. all the other terms from the entropy are cancelled out since
                                                        %  they appear with opposite sigh in the second term)

    % Improve the sample by drawing zt~Q(zt|z) using HMC
    [zt, samples, extraOutputs] = mcmc.algorithm(z, pxz, mcmc.inargs{:});
    % Keep track of acceptance rate (for information purposes only)
    acceptHist = acceptHist + extraOutputs.acceptHist/iters;
    acceptRate = acceptRate + extraOutputs.accRate/iters;
    % Adapt the stepsize for HMC
    mcmc.inargs{1} = extraOutputs.delta;
    
    % Evaluate the densities and the gradient (second expectation)
    logpx_zt = pxz.logdensity(zt, pxz.inargs{:});
    diff = (zt - vardist.mu)./vardist.sigma;
    diff2 = diff.^2;
    f_zt = logpx_zt + 0.5*sum(diff2);
    precond_mu_second = (eta./vardist.sigma).*(f_zt - control_variate);
    precond_sigma_second = (-1./vardist.sigma + (eta.^2)./vardist.sigma).*(f_zt - control_variate);
    
    % Evaluate the gradient (third expectation)
    precond_mu_third = diff./vardist.sigma;
    precond_sigma_third = diff2./vardist.sigma;
    
    % Total gradient
    grad_mu = precond_mu_first - precond_mu_second + precond_mu_third;
    grad_sigma = precond_sigma_first - precond_sigma_second + precond_sigma_third;

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

    % Stochastic divergence
    stochasticDiv(it) = -first_term(it) + mean(f_zt);
    entropy = 0.5*dim_z*log(2*pi) + sum(log(vardist.sigma),2) + dim_z/2;
    ELBO_q(it) = mean(logpxz) + mean(entropy);
    expLogLik_qt(it) = mean(logpx_zt);
    if mod(it,1000) == 0
        fprintf('Iter=%d, Stoch VCD=%f, ELBO_q=%f, stoch LogLik_qt=%f\n', it, stochasticDiv(it), ELBO_q(it), expLogLik_qt(it));
    end

    % Update the control variate
    if use_control_variate == 1
        control_variate = theta*control_variate + (1-theta)*f_zt;
    end

    % Decrease learning rates
    if mod(it, ReducedEvery) == 0
        rhosigma = rhosigma*ReducedBy;
        rhomu = rhomu*ReducedBy;
    end
%   
end
timetaken = toc; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Also fit the target with standard VI

vardistVCD = vardist;
vardist = vardistInit;

% Algorithm parameters
iters = 20000;
ELBO_q = zeros(1, iters);
rhomu = 0.01;
rhosigma = 0.005;
ReducedBy = 0.9;
ReducedEvery = 2000;
TT = 1;
kappa0 = 0.1;
Gt_mu = 0.01*ones(size( vardist.mu ));
Gt_sigma = 0.01*ones(size( vardist.sigma ));

tic;
for it=1:iters
%
    eta = randn(1,dim_z);
    z = vardist.mu + vardist.sigma.*eta;
    [logpxz, gradz] = pxz.logdensity(z, pxz.inargs{:});
    grad_mu = gradz;                                % grad assuming analytic entropy
    grad_sigma = gradz.*eta + 1./vardist.sigma;     % grad assuming analytic entropy

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

    entropy = 0.5*dim_z*log(2*pi) + sum(log(vardist.sigma),2) + dim_z/2;
    ELBO_q(it) = mean(logpxz) + mean(entropy);
    if mod(it,1000) == 0
        fprintf('Iter=%d, ELBO_q=%f\n', it, ELBO_q(it));
    end

    if mod(it, ReducedEvery) == 0
        rhosigma = rhosigma*ReducedBy;
        rhomu = rhomu*ReducedBy;
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
name = [outdir dataName '_VCD'];
print('-depsc2', '-r300', name);
cmd = sprintf('epstopdf %s', [name '.eps']);
system(cmd);

% Plot the densities
figure;
hold on;
x = linspace(xMin,xMax, 300);
y = linspace(yMin,yMax, 300);
[X,Y] = meshgrid(x,y);
Z = zeros(length(x), length(y));
ZvarSimple = zeros(length(x), length(y));
ZvarVCD = zeros(length(x), length(y));
for i=1:length(x)
    for j=1:length(y)
       Z(i,j) = pxz.logdensity([x(i), y(j)], pxz.inargs{:});
       ZvarSimple(i,j) = logdensityGaussian([x(i), y(j)], vardist.mu, diag(vardist.sigma));
       ZvarVCD(i,j) = logdensityGaussian([x(i), y(j)], vardistVCD.mu, diag(vardistVCD.sigma));
    end
end
[~, h_target] = contour(X,Y,exp(Z)', 4, 'Color', my_green, 'Linewidth', 1.2);
[~, h_standard] = contour(X,Y,exp(ZvarSimple)', 3, 'Color', my_blue, 'Linewidth', 1.2);
[~, h_newdiv] = contour(X,Y,exp(ZvarVCD)', 3, 'Color', my_red, 'Linewidth', 1.2);
box on;
name = [outdir dataName '_VCDvsStandardVI'];
print('-depsc2', '-r300', name);
cmd = sprintf('epstopdf %s', [name '.eps']);
system(cmd);
% Also export as pdf
set(gca, 'box', 'off');
figurepdf(3, 2.33);
print('-dpdf', [name '.pdf']);
