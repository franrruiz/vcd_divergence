function out = grad_elu(A) 
%

mask = (A>0); 
out = mask + exp(A).*(1-mask);  


