function out = grad_tanh(A) 
%

Z = tanh(A); 
out = 1 - Z.^2; 

