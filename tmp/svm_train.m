function [output] =svm_train(data,n_class,gamma,c,func)  
%多类别的SVM训练器  
if strcmp(func,'gauss')
    for i=1:n_class-1  
        for j=i+1:n_class  
            X=[data(5*(i-1)+1:5*i,:);data(5*(j-1)+1:5*j,:)];  
            Y=[ones(5,1);zeros(5,1)];  
            output{i}{j}=svmtrain(X,Y,'Kernel_Function',@(X,Y) kfun_rbf(X,Y,gamma,func),'boxconstraint',c);  
        end  
    end
elseif strcmp(func,'linear')
    for i=1:n_class-1  
        for j=i+1:n_class  
            X=[data(5*(i-1)+1:5*i,:);data(5*(j-1)+1:5*j,:)];  
            Y=[ones(5,1);zeros(5,1)];  
            output{i}{j}=svmtrain(X,Y,'Kernel_Function','linear');  
        end  
    end
end
end