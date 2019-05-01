function out = elu(A) 
%

mask = (A>0); 
out = A.*mask + (exp(A)-1).*(1-mask);  
