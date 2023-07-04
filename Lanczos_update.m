function [U, V, T, theta] = Lanczos_update(X, V, T, msteps)
%% function [U, V, lam,res] = Lanczos_update(X, V, T, msteps)
%% --------------------computes very approximate eigenvectors
%%                     via lanczos
%% X      = data matrix
%% msteps = number of lanczos steps
%% T and V = previous Tridiagonal matrix and its eigenvectors
%% RETURN
%% U = set of approximate eigenvectors.
%%  T and V = updated Tridiagonal matrix and its eigenvectors
%% lam    = set of msteps approximate eigenvectors == the first 5
%%          nev of these are the wanted eigenvalues
%% res    = residual norm information for each lambda in lam
%% --------------------pre-allocating
n = size(X,2);
m = msteps+1;

k1=size(V,2);
%%-------------------- initializations
v  = V(:,end);
beta = 0;
vold = v;
%%-------------------- for stopping --
orthTol = 1.e-08;
wn = 0.0 ;
%%-------------------- main
for k=1:msteps
    w = X*(X'*v);
    w = w - beta*vold ;
    alpha = w'*v;
    wn = wn + alpha*alpha;
    T(k1+k-1,k1+k-1) = alpha;
    w = w - alpha*v;
    %%-------------------- full reorthogonalization
    t = V'*w;
    
    w = w - V* t;
    beta = w'*w;
    if (beta*k < orthTol*wn)
        break
    end
    %%-------------------- for orth. test
    wn   = wn+2.0*beta;
    beta = sqrt(beta) ;
    vold = v;
    v = w/ beta;
    V(:,k1+k) = v;
    T(k1+k-1,k+k1) = beta;
    T(k+k1,k1+k-1) = beta;
end
%%--------------------
if length(T)<k1+k
    [U,theta] = eig(T(1:k1+k-1,1:k1+k-1));
else
    [U,theta] = eig(T(1:k1+k,1:k1+k));
end
[theta, ~] = sort(diag(theta),'descend');
