function [mean_px, px] = compute_llh_vae_explicit(S, pxz, vardist, data)
% Compute an importance sampling estimate of the log-evidence on test
%  S: number of samples for the importance sampling approximation
%  vardist: variational distribution
%  data: struct containing the data

px = zeros(data.test.N,1);
pxz.inargs{1} = pxz.vae;
for ns=1:data.test.N
    % Sample z
    Xinput = data.test.Xinput(ns,:);
    netMu = netforward(vardist.netMu, Xinput);
    netSigma = netforward(vardist.netSigma, Xinput);
    dim_z = size(netMu{1}.Z,2);
    eta = randn(S, dim_z);
    z = bsxfun(@plus, netMu{1}.Z, bsxfun(@times, 1.2*netSigma{1}.Z, eta));
    
    % Gaussian log q(z)
    logq = -0.5*dim_z*log(2*pi) - sum(log(1.2*netSigma{1}.Z)) - 0.5*sum(eta.^2, 2);
    
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
