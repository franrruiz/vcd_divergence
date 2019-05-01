function out = softplus(A) 
%

m = max(0,A);
out = m + log( exp(-m) + exp(A-m) );