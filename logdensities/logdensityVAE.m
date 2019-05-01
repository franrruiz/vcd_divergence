function [out, gradz, gradW, gradb] = logdensityVAE(z, net) 
%

% Pass the minibatch latent variables through the network 
net = netforward(net, z); 

% Outputs of the neural net
D = size(net{1}.Z, 2);
k = size(z,2); 

% Compute the log likelihood  (per data point in the minibatch)
switch net{1}.lik
%    
  case 'Gaussian'  % suitable for real-valued outputs 
    beta = exp(net{1}.logbeta);  
    term1 = - 0.5*sum( (net{1}.Z - net{1}.outData).^2, 2);
    out = beta*term1 - 0.5*D*log(2*pi) + 0.5*D*net{1}.logbeta;
    %grad_logbeta = beta*term1 + 0.5*N*D; 
  case 'Categorical' % categorical output data          
    M = max(net{1}.Z, [], 2);
    out = sum( net{1}.outData.*net{1}.Z, 2) - M  - log(sum(exp(bsxfun(@minus, net{1}.Z, M)), 2));
  case 'Bernoulli' % binary output data 
    out = sum(net{1}.outData.*log(net{1}.Z + 1e-50) + (1-net{1}.outData).*log(1 - net{1}.Z + 1e-50), 2);
  case 'Poisson' % Poisson assuming the intensity is based on softplus_threshold activation
    out = sum( net{1}.outData.*log(net{1}.Z) - net{1}.Z - net{1}.logfactX, 2);
end

% standard normal prior over the latent variables 
out = out - 0.5*k*log(2*pi) - 0.5*sum((z.^2),2); 


if nargout > 1 
%     
   % Error backpropagation for computing the derivatives wrt weights of the
   % network and the input (latent variables) 
   if strcmp(net{1}.lik,'Gaussian')
     Deltas = beta*(net{1}.outData - net{1}.Z);
   elseif strcmp(net{1}.lik,'Poisson')
     % it assumes the intensity is based on the softplus_threshold activation
     ee = 1e-4;
     Deltas = (net{1}.outData./net{1}.Z - 1).*sigmoid(net{1}.A-ee);
   else
     Deltas = net{1}.outData - net{1}.Z;
   end
   [gradW, gradb, gradz] = netbackpropagation(net, Deltas, 1);
      
   gradz = gradz - z;
%   
end
