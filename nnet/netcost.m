function [out, gradW, gradb] = netcost(X, T, net) 
%

net = netforward(net, X); 

[N D] = size(net{1}.Z);

% Compute the log likelihood  
switch net{1}.lik
%    
  case 'Gaussian'  % Gaussian log likelihood
    beta = exp(net{1}.logbeta);  
    term1 = - 0.5*sum(sum((net{1}.Z - T).^2));
    out = beta*term1 - 0.5*N*D*log(2*pi) + 0.5*N*D*net{1}.logbeta; 
    %grad_logbeta = beta*term1 + 0.5*N*D; 
 case 'logistic'  % logistic regression log likelihood     
    YF = - T.*net{1}.A;
    M = max(0,YF);
    out = - sum( sum( M + log( exp(-M) + exp( YF - M )) ) ); 
  case 'softmax' % log likelihood for multi-class classification         
    M = max(net{1}.A, [], 2);
    out = sum(sum( T.*net{1}.A )) - sum(M)  - sum(log(sum(exp(net{1}.A - M*ones(1, D)), 2)));
  case 'InfCrossEntropy'
    out = sum(sum( T.*log(net{1}.Z + 1e-50) + (1-T).*log(1 - net{1}.Z + 1e-50)));
end


if nargout > 1
%    
   % Error backpropagation for computing the derivatives wrt weights of the  network
   if strcmp(net{1}.lik,'Gaussian')
      Deltas = beta*(T - net{1}.Z);
   else
      Deltas = T - net{1}.Z;
   end
   [gradW, gradb] = netbackpropagation(net, Deltas, 1);      
%   
end
