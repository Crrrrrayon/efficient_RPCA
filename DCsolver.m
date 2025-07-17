function [X, T] = DCsolver(D, rho, T0, a, r)
[U, S, V] = svd(D, 'econ');
S_vec = diag(S);  
k = length(S_vec);

if length(T0) < k
    T0 = [T0; zeros(k - length(T0), 1)];
elseif length(T0) > k
    T0 = T0(1:k);
end

% DCInner
for t = 1:100
    [X, T1] = DCInner(S_vec, rho, T0, a, U, V, r);
    err = sum((T1 - T0).^2);
    if err < 1e-6
        break
    end
    T0 = T1;
end
T = T1;
end


function [X, t] = DCInner(S_vec, rho, J, epsilon, U, V, r)
lambda = 1/(2*rho);
k = length(S_vec);  

if length(J) < k
    J = [J; zeros(k - length(J), 1)];
elseif length(J) > k
    J = J(1:k);
end

grad = (exp(epsilon).*epsilon)./(epsilon + J).^2;

t = zeros(k, 1);
for i = 1:k
    if i <= min(r, k)  
        t(i) = max(S_vec(i) - lambda*grad(i), 0);
    else
        t(i) = 0;
    end
end

X = U * diag(t) * V';
end
