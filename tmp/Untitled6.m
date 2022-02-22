data_process_method = 'mean';
%gamma = [0.001;0.01;0.1;1;10;100;1000];
%c = [0.001;0.01;0.1;1;10;100;1000];
gamma = [0.1;1;10];
c = [0.1;1;10];

[acc0,acc2] = test_acc('mean','gauss',gamma,c);
%[acc11,acc12] = test_acc('mean','linear',gamma,c);
%[acc21,acc22] = test_acc('lbp','linear',gamma,c);


%[acc201,acc202] = test_acc2('mean','gauss',gamma,c);
% [acc211,acc212] = test_acc2('mean','linear',gamma,c);
% [acc221,acc222] = test_acc2('lbp','linear',gamma,c);

function [acc0,acc1]=test_acc(method,func,gamma,c)
if strcmp(func,'linear')
    [acc0,acc1] = main_func(method,0,0,func);
else
    acc0 = zeros(size(gamma,1),size(c,1));
    acc1 = zeros(size(gamma,1),size(c,1));
    for i = 1:size(gamma,1)
        for j = 1:size(c,1)
            [acc0(i,j),acc1(i,j)] = main_func(method,gamma(i),c(j),func);
        end
    end
end
end

function [acc0,acc1]=test_acc2(method,func,gamma,c)
if strcmp(func,'linear')
    [acc0,acc1] = main_func2(method,0,0,func);
else
    acc0 = zeros(size(gamma,1),size(c,1));
    acc1 = zeros(size(gamma,1),size(c,1));
    for i = 1:size(gamma,1)
        for j = 1:size(c,1)
            [acc0(i,j),acc1(i,j)] = main_func2(method,gamma(i),c(j),func);
        end
    end
end
end