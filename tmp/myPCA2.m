function [pca,V] = myPCA2(A,k,mA)
    [m,n] = size(A);
    %mA = mean(A,1);
    A_centered = A - ones(m,1)*mA;
    C = cov(A_centered);
    [V,D] = eigs(C,k);
    %fprintf("V = ");disp(V);fprintf("\nD = ");disp(D);fprintf("\n");
    pca = (A-mA)*V;
end