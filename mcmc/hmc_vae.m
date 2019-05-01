function [z, samples, extraOutputs] = hmc(current_q, log_pxz, epsilon, Burn, T, adapt, L)

[N n] = size(current_q);

acceptHist = zeros(N, Burn+T); 
logpxzHist = zeros(N, Burn+T);
% collected samples 
samples = zeros(N, n, T); 

if (Burn+T) == 0 
    z = current_q; 
    extraOutputs.delta = delta; 
    extraOutputs.accRate = 0;
    return;
end           
                      
eta = 0.01;
opt = 0.9;
cnt = 0; 
for i=1:(Burn + T)
%      
    q = current_q;
    p = randn(N,n); 

    current_p = p;

    % Make a half step for momentum at the beginning
    [logpxz, gradz] = log_pxz.logdensity(q, log_pxz.inargs{:});
    current_U = - logpxz;
    grad_U = - gradz;
    p = p - epsilon*grad_U/2;

    % Alternate full steps for position and momentum
    for j=1:L
    %
        % Make a full step for the position
        q = q + epsilon*p;
        % Make a full step for the momentum, except at end of trajectory
        if (j~=L) 
           [logpxz, gradz] = log_pxz.logdensity(q, log_pxz.inargs{:}); 
           proposed_U = - logpxz;
           grad_U = - gradz;
           p = p - epsilon*grad_U;
        end
    end

    % Make a half step for momentum at the end.
    [logpxz, gradz] = log_pxz.logdensity(q, log_pxz.inargs{:});
    proposed_U = - logpxz;
    grad_U = - gradz;
    p = p - epsilon*grad_U/2;
    % Negate momentum at end of trajectory to make the proposal symmetric
    p = -p;

    % Evaluate potential and kinetic energies at start and end of trajectory
    %current_U = U(current_q);
    current_K = sum(current_p.^2, 2)/2;
    %proposed_U = U(q);
    proposed_K = sum(p.^2, 2)/2; 
    % Accept or reject the state at end of trajectory, returning either
    % the position at the end of the trajectory or the initial position
    accept = (rand(N,1) < exp(current_U-proposed_U+current_K-proposed_K));
    
    acceptHist(:, i) = accept;   
   
    ind = find(accept==1);
    if length(ind) > 0 
      current_q(ind,:) = q(ind,:);
      current_U(ind) = proposed_U(ind);
    end
       
    % Adapt step size only during burn-in. After that
    % collect samples  
    if (i <= Burn) && (adapt == 1) 
       epsilon = epsilon + eta*((mean(accept) - opt)/opt)*epsilon;
    else
       cnt = cnt + 1;
       samples(:,:,cnt) = current_q;
    end
    logpxzHist(:,i) = - current_U;
end

z = current_q;
extraOutputs.logpxzHist = logpxzHist; 
extraOutputs.acceptHist = acceptHist;
extraOutputs.delta = epsilon; 
extraOutputs.accRate = mean(acceptHist, 2);
