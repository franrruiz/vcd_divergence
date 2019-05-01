function [mean_px, px] = compute_llh_vae_explicit_useHMC(S, pxz, vardist, data, mcmc, bb, ss, flag_sigma_proposal)
% Compute an importance sampling estimate of the log-evidence on test using
% MCMC to build a proposal distribution
%  S: number of samples for the importance sampling approximation
%  vardist: variational distribution
%  data: struct containing the data
%  mcmc: struct with the MCMC parameters
%  bb: Number of burn-in iterations for MCMC
%  ss: Number of sampling iterations for MCMC
%  flag_sigma_proposal: If 0, set the proposal std according to the std
%                       given by the amortization network. Otherwise, set
%                       the proposal std according to the empirical std of
%                       the MCMC samples
% 

px = zeros(data.test.N,1);
pxz.inargs{1} = pxz.vae;
mcmc.inargs{2} = bb;
mcmc.inargs{3} = ss;

% Obtain the variational mean and std
netMu = netforward(vardist.netMu, data.test.Xinput);
netSigma = netforward(vardist.netSigma, data.test.Xinput);
dim_z = size(netMu{1}.Z, 2);

% Start MCMC chain at the variational mean
pxz.inargs{1}{1}.outData = data.test.X;
if(isfield(data.test, 'logfactX'))
    pxz.inargs{1}{1}.logfactX = data.test.logfactX;
end
[~, zt_samples, ~] = mcmc.algorithm(netMu{1}.Z, pxz, mcmc.inargs{:});

% Obtain the parameters of the Gaussian proposal
proposal_mu = mean(zt_samples, 3);
if(flag_sigma_proposal)
    proposal_sigma = 1.2 * std(zt_samples, [], 3);
    proposal_sigma(proposal_sigma<1e-4) = 1e-4;     % for numerical stability
else
    proposal_sigma = 1.2*netSigma{1}.Z;
end

% For each instance
for ns=1:data.test.N
    % Sample z
    eta = randn(S, dim_z);
    z = bsxfun(@plus, proposal_mu(ns,:), bsxfun(@times, proposal_sigma(ns,:), eta));
    
    % Gaussian log q(z)
    logq = -0.5*dim_z*log(2*pi) - sum(log(proposal_sigma(ns,:))) - 0.5*sum(eta.^2, 2);
    
    % Evaluate the log-joint
    pxz.inargs{1}{1}.outData = repmat(data.test.X(ns,:), [S 1]);
    if(isfield(data.test, 'logfactX'))
        pxz.inargs{1}{1}.logfactX = repmat(data.test.logfactX(ns,:), [S 1]);
    end
    logjoint = pxz.logdensity(z, pxz.inargs{:});
    
    % Importance sampling term
    px(ns) = logsumexp(logjoint - logq, 1) - log(S);
end
mean_px = mean(px);
