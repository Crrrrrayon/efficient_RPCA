function [rank_est, info] = KGDE(X, max_rank, opts)
%KGDE Estimate matrix rank using Block Krylov accelerated GDE.

if nargin < 2 || isempty(max_rank)
    max_rank = min(64, min(size(X)));
end
if nargin < 3
    opts = struct();
end

%% Step 1: Prepare the working matrix and search settings
start_time = tic;
[m, n] = size(X);
max_rank = min(max_rank, min(m, n));
side = option(opts, 'covariance_side', 'smaller');

if strcmp(side, 'left') || (strcmp(side, 'smaller') && m <= n)
    A = X;
    side = 'left';
else
    A = X';
    side = 'right';
end

search_rank = min(max(option(opts, ...
    'initial_search_rank', 2), 1), max_rank);
q = option(opts, 'krylov_steps', 3);
seed = option(opts, 'seed', 1);

search_trace = [];
rank_trace = [];

while true
    %% Step 2: Build the dominant subspace with Block Krylov iteration
    [U, theta, bki_info] = block_krylov(A, search_rank, q, seed);

    %% Step 3: Form the covariance factor used by GDE
    F = U * diag(sqrt(theta));

    %% Step 4: Apply the paper GDE rank decision
    [candidate, gde_info] = factor_gde(F);

    search_trace(end+1, 1) = search_rank; %#ok<AGROW>
    rank_trace(end+1, 1) = candidate; %#ok<AGROW>

    %% Step 5: Stop or enlarge the Krylov search dimension
    inside_search = ~isempty(gde_info.first_negative_index) ...
        && gde_info.decision_index < search_rank;
    if inside_search || search_rank == max_rank
        break;
    end
    search_rank = min(2 * search_rank, max_rank);
end

rank_est = candidate;
if inside_search
    status = 'rank_found';
else
    status = 'maximum_search_rank_reached';
end

%% Step 6: Return reusable left and right singular subspaces
[left_basis, right_basis, singular_values] = ...
    singular_subspaces(X, U, theta, rank_est, side);

info = struct( ...
    'rank', rank_est, ...
    'status', status, ...
    'covariance_side', side, ...
    'final_search_rank', search_rank, ...
    'maximum_search_rank', max_rank, ...
    'krylov_steps', q, ...
    'seed', seed, ...
    'search_rank_trace', search_trace, ...
    'candidate_rank_trace', rank_trace, ...
    'covariance_factor', F, ...
    'left_basis', left_basis, ...
    'right_basis', right_basis, ...
    'singular_values', singular_values, ...
    'bki_info', bki_info, ...
    'gde_info', gde_info, ...
    'total_time', toc(start_time));
end


function [U, theta, info] = block_krylov(X, k, q, seed)
%BLOCK_KRYLOV Approximate the dominant covariance eigenspace.

[m, n] = size(X);

% Create the random starting block.
rng(seed, 'twister');
Omega = randn(n, k);

% Expand and collect all Krylov blocks.
K = zeros(m, min(m, (q + 1) * k));
Y = X * Omega;
first = 1;

for j = 0:q
    scale = norm(Y, 'fro');
    if scale == 0
        break;
    end
    Y = Y / scale;

    last = min(first + k - 1, size(K, 2));
    width = last - first + 1;
    if width <= 0
        break;
    end
    K(:, first:last) = Y(:, 1:width);
    first = last + 1;

    if j < q
        Y = X * (X' * Y);
    end
end

% Orthonormalize the complete Krylov subspace.
K = K(:, 1:first-1);
[Q, ~] = qr(K, 0);

% Extract the dominant directions by Rayleigh--Ritz.
B = Q' * (X * (X' * Q));
B = full((B + B') / 2);

[W, D] = eig(B);
[all_theta, order] = sort(real(diag(D)), 'descend');
W = W(:, order);

theta = max(all_theta(1:k), 0);
U = Q * W(:, 1:k);

info = struct( ...
    'target_rank', k, ...
    'krylov_steps', q, ...
    'krylov_dimension', size(K, 2), ...
    'ritz_values', all_theta, ...
    'orthogonality_error', norm(U' * U - eye(k), 'fro'));
end


function [rank_est, info] = factor_gde(F)
%FACTOR_GDE Apply the GDE decision without forming a large covariance.

[k, width] = size(F);

% Obtain the leading-block spectrum and disk radii.
F1 = F(1:k-1, :);
f = F(k, :)';
[~, S, V] = svd(full(F1), 'econ');
s = real(diag(S));
count = min(numel(s), k-1);
s = s(1:count);
V = V(:, 1:count);

lambda = zeros(k-1, 1);
raw_radii = zeros(k-1, 1);
lambda(1:count) = s .^ 2;
raw_radii(1:count) = abs(s .* (V' * f));

% Shrink the disk radii.
scaled_radii = abs(lambda) .* raw_radii / norm(lambda);

% Evaluate the GDE score for every candidate rank.
scores = zeros(k-2, 1);
adjustment = zeros(k-2, 1);
for t = 1:k-2
    tail_norm = norm(lambda(t:k-1));
    adjustment(t) = 2 * abs(lambda(t+1)) / tail_norm;
    scores(t) = scaled_radii(t) ...
        - adjustment(t) * mean(scaled_radii);
end

% Return the rank before the first negative score.
first_negative = find(scores(2:end) < 0, 1);
if ~isempty(first_negative)
    first_negative = first_negative + 1;
end
if isempty(first_negative)
    decision_index = k - 2;
    status = 'no_negative_score';
else
    decision_index = first_negative;
    status = 'rank_found';
end
rank_est = decision_index - 1;

info = struct( ...
    'rank', rank_est, ...
    'status', status, ...
    'factor_width', width, ...
    'leading_eigenvalues', lambda, ...
    'raw_radii', raw_radii, ...
    'scaled_radii', scaled_radii, ...
    'scores', scores, ...
    'adjustment_factors', adjustment, ...
    'first_negative_index', first_negative, ...
    'decision_index', decision_index);
end


function [UL, VR, singular_values] = ...
        singular_subspaces(X, U, theta, rank_est, side)
%SINGULAR_SUBSPACES Build reusable singular vectors after rank selection.

UL = zeros(size(X, 1), 0);
VR = zeros(size(X, 2), 0);
singular_values = zeros(0, 1);

if ~isfinite(rank_est) || rank_est < 1
    return;
end

r = min(rank_est, numel(theta));
singular_values = sqrt(theta(1:r));
safe_values = max(singular_values, eps(class(singular_values)));

if strcmp(side, 'left')
    UL = U(:, 1:r);
    VR = (X' * UL) ./ safe_values';
else
    VR = U(:, 1:r);
    UL = (X * VR) ./ safe_values';
end
end


function value = option(opts, name, default_value)
if isfield(opts, name) && ~isempty(opts.(name))
    value = opts.(name);
else
    value = default_value;
end
end
