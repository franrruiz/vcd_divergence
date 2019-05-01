function [dW, db, dX] = netbackpropagation(net, Deltas, flag)
%
%

if nargin == 2
    flag = 0;
end

% when flag == 1 ignore the gradient from the last transformation 
% (might have been incorporated from outside together with the cost function for numercial stability)
if flag == 0
   Deltas = Deltas.*net{1}.grad_actfunc(net{1}.A);
end

L = length(net)-2;
for layer=1:L+1
%    
  dW{layer} = net{layer+1}.Z'*Deltas;
  db{layer} = sum(Deltas, 1);  
  if layer < (L+1)
     Deltas = Deltas*net{layer}.W';
     Deltas = Deltas.*net{layer+1}.grad_actfunc(net{layer+1}.A);
  end
%
end

if nargout == 3 
   % Output also the derivatives wrt the inputs of the net if needed 
   dX = Deltas*net{L+1}.W';
end
