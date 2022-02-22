function [accuracy1,accuracy2] = main_func(data_process_method,gamma,c,func)
npersons=40;%ѡȡ40���˵���  
k=20;%��ά��1ά 
%gamma=0.1; %svm����
%c=10;  %svm����
global imgrow;  
global imgcol;  
imgrow=112;  
imgcol=92;  
%data_process_method = 'mean';
[f0_matrix, RealClassTrain] =ReadFace(npersons,0);%��ȡѵ������
f_matrix = data_preprocess(f0_matrix,data_process_method);

ma = mean(f_matrix);
[pcaface,V]=myPCA2(f_matrix,k,ma);%���ɷַ�����������ȡ  
   
multiSVMstruct=svm_train(pcaface,npersons,gamma,c,func);  

class= multi_SVM(pcaface,multiSVMstruct,npersons);
% 
accuracy1=sum(class==RealClassTrain)/length(class);  
%msgbox(['ѵ��׼ȷ�ʣ�',num2str(accuracy*100),'%��'])
%fprintf("ѵ��׼ȷ��:%d%%\n",accuracy*100);

[testface0,realclass]=ReadFace(npersons,1);
testface = data_preprocess(testface0,data_process_method);

testface2 = cal_latent('pca',testface,V,ma);
class= multi_SVM(testface2,multiSVMstruct,npersons);  

accuracy2=sum(class==realclass)/length(class);  
%msgbox(['ʶ��׼ȷ�ʣ�',num2str(accuracy*100),'%��'])
%fprintf("ѵ��׼ȷ��:%.2f %% %.2f %%\n",accuracy1*100,accuracy2*100);

function [output] = cal_latent(func_name,test_data,V,mA)
if strcmp(func_name,'pca')
    output = (test_data-mA)*V;
elseif strcmp(func_name,'lda')
    output = test_data*V;
end

end

end