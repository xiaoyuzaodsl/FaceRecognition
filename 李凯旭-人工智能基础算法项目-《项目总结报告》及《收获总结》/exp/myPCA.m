function [pca_matrix,V] = myPCA(A,k,mA)
    [m,n] = size(A);
    A_centered = A - ones(m,1)*mA;
    C = cov(A_centered);
    [V,D] = eigs(C,k);
    pca_matrix = (A-mA)*V;
end