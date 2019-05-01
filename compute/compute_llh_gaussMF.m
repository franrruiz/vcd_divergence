function [mean_px, px] = compute_llh_gaussMF(pxz, data)
% Compute the exact log-evidence on test
%  vardist: variational distribution
%  data: struct containing the data

D = size(pxz.vae{1}.W, 2);
s2_prior = exp(-pxz.vae{1}.logbeta);
aux_cov = pxz.vae{1}.W'*pxz.vae{1}.W + s2_prior*eye(D);
L = chol(aux_cov)';
diff = L\(data.test.X');
px = -0.5*D*log(2*pi) - sum(log(diag(L))) - 0.5*sum(diff'.^2, 2);
mean_px = mean(px);
