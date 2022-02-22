function [pca,V,mA] = myPCA(A,k)
    [m,n] = size(A);
    %fprintf("m = %d and n = %d\n",m,n);
    mA = mean(A,1);
    %disp(mA);
    A_centered = A - ones(m,1)*mA;
    %disp(A_centered);
    C = cov(A_centered);
    %C = A_centered * A_centered';
    %disp(C);
    [V,D] = eigs(C,k);
    %fprintf("V = ");disp(V);fprintf("\nD = ");disp(D);fprintf("\n");
    pca = (A-mA)*V;
end