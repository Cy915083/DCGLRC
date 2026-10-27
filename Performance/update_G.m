function G = update_G(Z, F, beta, lambda, k, Q, J, mu)
    
    V = length(Z);          % number of views
    [n, ~] = size(F);       % number of samples
    
    % ========== 1. Compute the original distance matrix D_orig = Σ_v ||z_i^(v) - z_j^(v)||² + λ * ||f_i - f_j||² ==========
    D_orig = zeros(n, n);
    
    % 1.1 Squared Euclidean distances of the self-representation matrix of each view
    for v = 1:V
        Zv = Z{v};                         % n×n
        % Column-wise sum of squares: fast computation of diag(Zv' * Zv)
        col_sq = sum(Zv.^2, 1);            % 1×n
        % Distance matrix: ||z_i - z_j||² = col_sq(i) + col_sq(j) - 2*(Zv'*Zv)
        Dv = bsxfun(@plus, col_sq', col_sq) - 2 * (Zv' * Zv);
        D_orig = D_orig + Dv;
    end
    
    % 1.2 Squared Euclidean distances between rows of F (each row is a c-dimensional sample)
    if ~isempty(F)
        row_sq = sum(F.^2, 2);              % n×1
        Df = bsxfun(@plus, row_sq, row_sq') - 2 * (F * F');
        D_orig = D_orig + lambda * Df;      % the λ * ||f_i - f_j||² term in the original formulation
    end
    
    % Set the diagonal to zero (distance to itself is 0)
    D_orig(1:n+1:end) = 0;
    
    % ========== 2. Construct the coefficient matrix C, where C_ij = (D_orig_ij + Q_ij - ρ J_ij) / (2(β + ρ/2)) ==========
    denom = 2 * (beta + mu / 2);           % denominator
    C = (D_orig + Q - mu * J) / denom;
    
    % ========== 3. For each sample i, independently solve g_i = argmin ∥g_i + c_i∥², s.t. g_i≥0, sum(g_i)=1 ==========
    G = zeros(n, n);
    
    for i = 1:n
        % Get the vector c_i corresponding to the i-th row
        c_i = C(i, :);
        c_i(i) = inf;               % exclude itself (ensure g_ii = 0)
        
        % Sort in ascending order to get sorted values and original indices
        [c_sorted, idx] = sort(c_i, 'ascend');
        
        % Determine the effective number of neighbors based on the preset neighbor count k (cannot exceed n-1)
        k_i = min(k, n-1);
        if k_i == 0
            % Degenerate to a uniform distribution (excluding itself)
            g_i = ones(1, n) / (n - 1);
            g_i(i) = 0;
            G(i, :) = g_i;
            continue;
        end
        
        % Take the (k_i+1)-th smallest value (note the sorted array has length n)
        c_k1 = c_sorted(k_i + 1);
        sum_c_k = sum(c_sorted(1:k_i));
        denom_k = k_i * c_k1 - sum_c_k;
        
        % Handle the case where the denominator is non-positive: decrease k_i until the denominator is positive
        if denom_k <= 0
            found = false;
            for temp_k = k_i-1 : -1 : 1
                c_k1_temp = c_sorted(temp_k + 1);
                sum_c_temp = sum(c_sorted(1:temp_k));
                denom_temp = temp_k * c_k1_temp - sum_c_temp;
                if denom_temp > 0
                    k_i = temp_k;
                    c_k1 = c_k1_temp;
                    sum_c_k = sum_c_temp;
                    denom_k = denom_temp;
                    found = true;
                    break;
                end
            end
            if ~found
                % No valid k_i can be found; use uniform allocation over the first min(k, n-1) neighbors
                g_i = zeros(1, n);
                n_nbr = min(k, n-1);
                g_i(idx(1:n_nbr)) = 1 / n_nbr;
                g_i(i) = 0;
                G(i, :) = g_i;
                continue;
            end
        end
        
        % Compute g_i according to the closed-form formula (29)
        g_i = zeros(1, n);
        for j = 1:k_i
            original_idx = idx(j);
            g_i(original_idx) = (c_k1 - c_sorted(j)) / denom_k;
        end
        g_i(i) = 0;                     % ensure self-similarity is 0
        
        % Normalize (for numerical stability)
        sum_g = sum(g_i);
        if sum_g > 0
            g_i = g_i / sum_g;
        else
            % Fallback when entirely zero: uniformly distribute over the first k_i neighbors
            g_i = zeros(1, n);
            g_i(idx(1:k_i)) = 1 / k_i;
            g_i(i) = 0;
        end
        
        G(i, :) = g_i;
    end
    
    % ========== 4. Post-processing: symmetrization, non-negativity, row normalization, zero diagonal ==========
    G = (G + G') / 2;               % enforce symmetry (undirected graph)
    G = max(G, 0);                  % ensure non-negativity
    G(1:n+1:end) = 0;               % re-ensure the diagonal is 0
    
    % Row normalization (to handle small numerical errors)
    row_sum = sum(G, 2);
    zero_rows = (row_sum == 0);
    if any(zero_rows)
        for i = find(zero_rows)'
            G(i, :) = 1 / (n - 1);
            G(i, i) = 0;
        end
        row_sum(zero_rows) = 1;
    end
    G = bsxfun(@rdivide, G, row_sum);
    
    % Optional: symmetrize once more to avoid asymmetry caused by row normalization
    G = (G + G') / 2;
    G(1:n+1:end) = 0;
end