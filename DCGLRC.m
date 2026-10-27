function [result] = DCGLRC(X,alpha,beta,gamma,lambda,k,gt)
% profile on
addpath('Performance');
epson = 1e-7;
Clus_num = size(unique(gt),1); % number of clusters
V = size(X,2); % number of views
N = size(X{1},2); % number of data points

% Initialize history record arrays
iter_history = [];
obj_history = [];
max_e1_history = [];
max_e2_history = [];

% Initialize X
for i=1:V
    X{i} = NormalizeFea(X{i},0);
end

% Initialize variables
S = zeros(N,N);
G = zeros(N,N);
F = orth(rand(N, Clus_num)); 
sX = [N,N,2];
q = zeros(N*N*2,1);
j = zeros(N*N*2,1);

for i = 1:V
    Z{i} = zeros(N,N);
    E{i} = zeros(size(X{i},1),N);
    % Lagrange multipliers
    Y{i} = zeros(size(X{i},1),N);
end

for a=1:2
    Q{a} = zeros(N,N);
    J{a} = zeros(N,N);
end

% Initialize weights
weight = 1/V*ones(1,V);

% Initialize penalty parameter
mu = 1;
mu_max = 10e10;

% Initialize eta
eta = 1.5;
iter = 0;
max_iter = 100;

%% iteration

while(iter < max_iter)
    G = (G+G')/2;
    D = diag(sum(G,2));
    Lg = D - G;
    
    %% == update Z{i} ==
    for i=1:V
        A1 = 2*Lg + 2*weight(1,i)*eye(N) + mu*X{i}'*X{i};
        A2 = 2*weight(1,i)*S + mu*X{i}'*(X{i}-E{i}+Y{i}/mu);
        Z{i} = A1\ A2;
    end

    %% == update F ==
    [V_F, ~] = eigs(Lg, Clus_num, 'smallestreal');
    F = V_F;

    %% == update S ==
    tmp1 = 0;
    tmp2 = 0;
    for i=1:V
        tmp1 = tmp1 +2* weight(1,i)*Z{i}-Q{1}+mu*J{1};
        tmp2 = tmp2 +2* weight(1,i)+mu*eye(N);
    end
    S_Star = tmp1 / (tmp2 + eps);
    temp_S = SimplexProj(S_Star);
    S = max(temp_S, 0);

    %% == update G ==
    G = update_G(Z, F, beta, lambda, k, Q{2}, J{2}, mu);

    %% == update E{i} ==
    E_concat = [];
    for v = 1:V
        M = X{v} - X{v} * Z{v} + Y{v} / mu;
        E{v} = solve_l1l2(M, alpha / mu);
        E_concat = [E_concat; E{v}];
    end

    %% == update J ==
    H_tensor = cat(3, S, G);
    Q_tensor = cat(3, Q{:,:});
    h = H_tensor(:);
    q = Q_tensor(:);  
    [j, ~] = wshrinkObj(h + 1/mu*q,gamma/mu,sX,0,3);
    J_tensor = reshape(j, sX);
    for a = 1:2
        J{a} = J_tensor(:,:,a);
        Q_tensor = reshape(q, sX);
        Q{a} = Q_tensor(:,:,a);
    end
    q = q + mu*(h - j);

    %% == Compute errors ==
    % Initialize error array
    e1 = zeros(1,V);
    for i = 1:V
        e1(i) = norm(X{i}-X{i}*Z{i}-E{i}, 'inf');
    end
    e2 = H_tensor - J_tensor;
    em1 = max(e1);
    em2 = max(abs(e2(:)));
    err = max([em1, em2]);

    %% == Check convergence ==
    if err < epson
        % fprintf('Convergence condition satisfied; iteration terminated.\n');
        break;
    end

    iter = iter + 1;
    
    %% == update mu ==
    mu = min(mu_max, mu*eta);
    
    %% == update Y{i}, Q ==
    for i = 1:V
        Y{i} = Y{i} + mu*(X{i}-X{i}*Z{i}-E{i});
    end
    q = q + mu*(h - j);
end
    
%% Final Clustering
G_final = (abs(G) + abs(G)') / 2;
G_bin   = G_final > 1e-12;

try
    cc = conncomp(graph(G_bin));
    n_comp = numel(unique(cc));
catch
    n_comp = -1;
end

if n_comp == Clus_num
    Clus = cc(:);
else
    Clus = SpectralClustering(G_final, Clus_num);
end

result = EvaluationMetrics(gt, Clus);

end