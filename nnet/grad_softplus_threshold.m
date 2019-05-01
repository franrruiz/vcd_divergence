function out = grad_softplus_threshold(A) 
%

ee = 1e-4;
out = sigmoid(A-ee);