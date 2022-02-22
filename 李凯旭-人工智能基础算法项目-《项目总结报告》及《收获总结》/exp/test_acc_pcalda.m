% data_method = 'raw';
% data_method = 'lbp';
data_method = 'uni_lbp';
%gamma = [0.001;0.01;0.1;1;10;100;1000];
%c = [0.001;0.01;0.1;1;10;100;1000];
gamma = [0.001;0.01;0.1];
%c = [0.001;0.01;0.1;1;10];
c = [0.01;0.1;1];
kernal_func = 'linear';
[acc_a1,acc_a2] = test_acc(data_method,kernal_func,gamma,c,'pca');
[acc_b1,acc_b2] = test_acc(data_method,kernal_func,gamma,c,'lda');
[acc_c1,acc_c2] = test_acc(data_method,kernal_func,gamma,c,'pcalda');

kernal_func = 'gauss';
% [acc_a1,acc_a2] = test_acc(data_method,kernal_func,gamma,c,'pca');
% [acc_b1,acc_b2] = test_acc(data_method,kernal_func,gamma,c,'lda');
% [acc_c1,acc_c2] = test_acc(data_method,kernal_func,gamma,c,'pcalda');


function [acc0,acc1]=test_acc(method,kernal_func,gamma,c,type)
if strcmp(kernal_func,'linear')
    if strcmp(type,'pca')
        [acc0,acc1] = pca_svm_all(method,0,0,kernal_func);        
    elseif strcmp(type,'lda')
        [acc0,acc1] = lda_svm_all(method,0,0,kernal_func);        
    elseif strcmp(type,'pcalda')
        [acc0,acc1] = pcalda_svm_all(method,0,0,kernal_func);
    end
else
    acc0 = zeros(size(gamma,1),size(c,1));
    acc1 = zeros(size(gamma,1),size(c,1));
    for i = 1:size(gamma,1)
        for j = 1:size(c,1)
            if strcmp(type,'pca')
                [acc0(i,j),acc1(i,j)] = pca_svm_all(method,gamma(i),c(j),kernal_func);        
            elseif strcmp(type,'lda')
                [acc0(i,j),acc1(i,j)] = lda_svm_all(method,gamma(i),c(j),kernal_func);        
            elseif strcmp(type,'pcalda')
                [acc0(i,j),acc1(i,j)] = pcalda_svm_all(method,gamma(i),c(j),kernal_func);
            end
        end
    end
end
end
