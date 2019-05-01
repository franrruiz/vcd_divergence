function out = softplus_threshold(A) 
%
ee = 1e-4;
out = zeros(size(A));
idxP = (A>=0);
idxN = (A<0);
out(idxP) = A(idxP) + log(1 + exp(ee-A(idxP)));
out(idxN) = ee + log(1 + exp(A(idxN)-ee));
