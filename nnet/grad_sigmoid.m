function out = grad_sigmoid(A) 
%

Z = sigmoid(A);
out = Z.*(1 - Z); 