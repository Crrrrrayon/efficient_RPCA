function [q ]= KGDE(X,  max_rank)
% Inputs:
% X: Input matrix.
% max_rank
% Output:
% q: Estimated rank.

[p,n] = size(X);  % get size
%% Main routine

v=randn(p,1); V=v/norm(v);  % crerate a random vector for Lanczos
T=[];

for k=1:max_rank
    [U, V, T, Theta] = Lanczos_update(X, V, T, k+1);   % get the next eigenpairs using the Lanczos algorithm
    % GDE
    R_k = V*diag(Theta)*U;
    RR = R_k'*R_k;
    [V_eig,D_eig]=eigs(RR);
    [D_sort,index] = sort(diag(D_eig),'descend');
    D_sort = diag(D_sort);
    V_sort = V_eig(:,index);
    
    R = V_sort*D_sort*V_sort';
    [M1,N1]=size(R);
    R_A_T = zeros(M1-1,N1-1);
    for p1=1:M1-1
        for q1=1:M1-1
            R_A_T(p1,q1)=R(p1,q1);
        end
    end
    [V_new,D_new]=eig(R_A_T);
    t=zeros(M1-1,1);
    T_new=[V_new t;t' 1];
    R1 = R;
    T1 = T_new;
    R_T=T1'*R1*T1;
    r_r_1 = zeros(1,N1-1);
    for i=1:N1-1
        r_r_1(i)=abs(R_T(i,N1));
    end
    D1=diag(D_new);
    
    D_seta_N=1/sqrt(D1'*D1);
    D_seta = D1*D_seta_N;
    % Shrink
    r_r = r_r_1';
    r_r = r_r.*D_seta;
    r=abs(r_r');
    
    if N1<=2
        j=2;
    else
        for j=1:N1-2
            D_M=abs(2*D1(j+1))/(sqrt(D1(j:N1-1)'*D1(j:N1-1)));
            m_seta_r=sum(r);
            D_MM = D_M*(m_seta_r/(N1-1));
            GDE_K(j)=r(j)-D_MM;
            if GDE_K(j)<0
                break;
            end
        end
    end
    if k>1 && j<k
        break;
    end
end
q=j-1;   % estimated rank


