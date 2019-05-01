function block = takeNextBatch(N, numBatch, opt_st, opt_perm)

persistent st;
persistent perm;

if(nargin>=3)
    st = opt_st;
elseif(isempty(st))
    st = 1;
end
if(nargin>=4)
    perm = opt_perm;
elseif(isempty(perm))
    perm = randperm(N);
end
    
% take a minibatch 
if (st+numBatch-1) <= N
    block = perm(st:st+numBatch-1);
    st = st+numBatch;
else
    block = perm(st:end);
    B = length(block);
    idx_diff = setdiff(1:N, block);
    idx_intersect = block;
    
    aux_idx = permute_vector(idx_diff);
    block = [block aux_idx(1:numBatch-B)];
    perm = [aux_idx(1:numBatch-B) permute_vector([idx_intersect aux_idx(numBatch-B+1:end)])];
    st = numBatch-B+1;
end
