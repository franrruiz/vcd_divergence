function dX = netbackpropagationGradofInput(net, Deltas, flag)
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
  if layer < (L+1)
     Deltas = Deltas*net{layer}.W';
     Deltas = Deltas.*net{layer+1}.grad_actfunc(net{layer+1}.A);
  end
%
end

% Output the derivatives wrt the inputs of the net 
dX = Deltas*net{L+1}.W';
