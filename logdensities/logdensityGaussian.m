function [out, gradz] =logdensityGaussian(x, mm, L)

aux = L\((x-mm)');
out = -0.5*length(x)*log(2*pi) - sum(log(diag(L))) - 0.5*(aux'*aux);

if nargout > 1
  gradz = L'\aux;
  gradz = - gradz';
end
