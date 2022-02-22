function [output] = calculate_embedding(func_name,test_data,V,mA)
if strcmp(func_name,'pca')
    output = (test_data-mA)*V;
elseif strcmp(func_name,'lda')
    output = test_data*V;
end
end