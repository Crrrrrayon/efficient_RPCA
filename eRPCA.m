function [ L, S] = eRPCA( D, para )
%
% Inputs:
% D : Observed matrix. Sum of underlying low rank matrix and underlying
%     sparse matrix.
% rankN : Estimated rank of underlying low rank matrix by KGDE.
%
% Outputs:
% L : Low rank part.
% S : Sparse part.
%
%

[m,n]     = size(D);

%% Default/Inputed parameters
max_iter  = 200;
tol       = 1e-7;
zeta_init = max(abs(D(:)));
mu        = 5;
con       = 4;
resample  = true;
eta = para.eta;
gamma = 0.02;

%% parameter setting
if isfield(para,'zeta_init')
    zeta_init = para.zeta_init;
    fprintf('zeta_init = %f set.\n', zeta_init);
else
    fprintf('using default zeta_init = %f.\n', zeta_init);
end

if isfield(para,'eta')
    eta = para.eta;
    fprintf('eta = %f set.\n', tol);
else
    fprintf('using default eta = %f.\n', eta);
end

if isfield(para,'mu')
    mu = para.mu;
    fprintf('mu = [%f,%f] set.\n', mu(1), mu(end));
else
    fprintf('using default mu = [%f,%f].\n', mu, mu);
end

if isfield(para,'max_iter')
    max_iter = para.max_iter;
    fprintf('max_iter = %d set.\n', max_iter);
else
    fprintf('using default max_iter = %d.\n', max_iter);
end

if isfield(para,'tol')
    tol= para.tol;
    fprintf('tol = %e set.\n', tol);
else
    fprintf('using default tol = %e.\n', tol);
end

if isfield(para,'con')
    con = para.con;
    fprintf('sample const = %d set. \n',con);
else
    fprintf('using default sample const = %d. \n',con);
end

if isfield(para,'resample')
    resample = para.resample;
    fprintf('resample = %d set. \n',resample);
else
    fprintf('using default resample = %d. \n',resample);
end

err    = -1*ones(max_iter,1);
timer  = zeros(max_iter,1);

tic
[m,n] = size(D);
% Estimate Rank by KGDE
m1 = 10; % parameter
rank_N(1) = KGDE(D,m1,min(m,n));
rankN = rank_N(1);

siz_col = ceil(con*rankN);
siz_row = ceil(con*rankN);

if ~resample
    rows = randi(m,1,siz_row);
    cols = randi(n,1,siz_col);
    rows = unique(rows);
    cols = unique(cols);
    D_cols = D(:,cols);
    D_rows = D(rows,:);
    norm_of_D = (norm(D_rows, 'fro')+ norm(D_cols, 'fro'));
end
init_timer = toc;

lambda=1e-5;
rate=1.05;
mu2 = 1e-4;


%% main algorithm
for t = 1 : max_iter
    tic;
    
    %% Resample
    if resample
        rows = randi(m,1,siz_row);
        cols = randi(n,1,siz_col);
        rows = unique(rows);
        cols = unique(cols);
        D_cols = D(:,cols);
        D_rows = D(rows,:);
        norm_of_D = (norm(D_rows, 'fro')+ norm(D_cols, 'fro'));
    end
    
    %% update S
    if t == 1
        zeta = zeta_init;
        L_cols = zeros(size(D_cols));
        L_rows = zeros(size(D_rows));
        S_cols = wthresh( D_cols,'h',zeta);
        S_rows = wthresh( D_rows,'h',zeta);
    else
        zeta = eta * zeta;
        L_cols = C*pinv_U*(R(:,cols));
        L_rows = (C(rows,:))*pinv_U*R;
        S_rows = wthresh( D_rows-L_rows,'h',zeta);
        S_cols = wthresh( D_cols-L_cols,'h',zeta);
    end
    
    %% Update L = C * pinv_U * R
    if t == 1
        sig1 = zeros(min(size(S_cols)),1);
        sig2 = zeros(min(size(S_rows)),1);
    else
        sig1 = [sig1(1:rankN),zeros(1,size(S_cols,2)-rankN)];
        sig2 = [sig2(1:rankN),zeros(1,size(S_rows,1)-rankN)];
    end
    
    % Estimate Rank per Iteration
%     rak(t+1) = rank(D_cols-S_cols);
%     rank_N(t+1) = KGDE(D_cols-S_cols,m1,min(m,n));
%     if rank_N==0
%         rank_N = 1;
%     end
%     rankN = rank_N(t+1);
%     disp(['Rank =',num2str(rankN),',iter = ',num2str(t)])
    
    [C,sig1] = DCsolver(D_cols-S_cols,mu2/lambda,sig1,gamma,rankN);
    [R,sig2] = DCsolver(D_rows-S_rows,mu2/lambda,sig2,gamma,rankN);
    
    MU = C(rows,:);
    [Uu,Su,Vu] = svd(MU);
    d = diag(Su);
    Su = diag(1./d(1:rankN));
    pinv_U = Vu(:,1:rankN)*Su*(Uu(:,1:rankN))';
    
    mu2 = mu2*rate;
    %% Stop Condition
    err(t) = (norm(D_rows-L_rows-S_rows, 'fro') + norm(D_cols-L_cols-S_cols, 'fro')) / norm_of_D;
    timer(t) = toc;
    
    if err(t) < tol
        fprintf('Total %d iteration, final error: %e, total time: %f  \n', t, err(t), sum(timer(timer>0)));
        timer(1) = timer(1) + init_timer;
        timer = timer(1:t);
        err = err(1:t);
        L = C * pinv_U * R;
        S = D - L;
        return;
    else
        fprintf('Iteration %d: error: %e, timer: %f \n', t, err(t), timer(t));
    end
    
end

fprintf('Maximum iterations reached, final error: %e.\n======================================\n', err(t));
timer(1) = timer(1) + init_timer;

end

