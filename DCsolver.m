function [ X,T ] = DCsolver(D,rho,T0,a,r)

[U,S,V] = svd(D,'econ');

for t = 1:100
    
    [ X,T1 ] = DCInner(S,rho,T0,a,U,V,r);
    err = sum((T1-T0).^2);
    if err < 1e-6
        break
    end
    T0 = T1;
end
T = T1;
end


function [ X,t ] = DCInner(S,rho,J,epislon,U,V,r)
lambda=1/2/rho;
S0 = diag(S);
grad=(exp(epislon).*epislon)./(epislon+J).^2;
for i=1:length(S0)
    if i<=r
        t(i)=max(S0(i)-lambda*grad(i),0);
    else
        t(i)=0;
    end
end
X=U*diag(t)*V';
end
