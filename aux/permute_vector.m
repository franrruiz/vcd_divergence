function y = permute_vector(x)

N = length(x);
aux = randperm(N);
y = x(aux);
