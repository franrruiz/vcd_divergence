function [out, gradz] = logdensityLogisticRegression(z, data, s2w) 
% z    : (1 x n) latent variable
% data : data struct (it contains 'data.N', 'data.D', 'data.X', 'data.Y')
% s2w  : prior variance on the regression coefficients

n = length(z);
Xz = data.X*z';
out = -0.5*n*log(2*pi) - 0.5*n*log(s2w) - 0.5*(z*z')/s2w ...
      + sum( logsigmoid(data.titleY.*Xz) );
if nargout > 1
    % Evaluate gradient of the model
    gradz  = - z/s2w + ...
             + (data.Y - sigmoid(Xz))'*data.X;
end
