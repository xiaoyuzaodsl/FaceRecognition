function [accuracy1,accuracy2] = pca_svm_all(data_process_method,gamma,c,func)
npersons=40;%ѡȡ40���˵���  
k=20;%��ά��1ά 
%gamma=0.1; %svm����
%c=10;  %svm����
global imgrow;  
global imgcol;  
imgrow=112;  
imgcol=92;  
%data_process_method = 'mean';
[f0_matrix, RealClassTrain] =read_face(npersons,0);%��ȡѵ������
f_matrix = data_process(f0_matrix,data_process_method);

ma = mean(f_matrix);
[pcaface,V]=myPCA(f_matrix,k,ma);%���ɷַ�����������ȡ  
   
multiSVMstruct=multi_svm_struct(pcaface,npersons,gamma,c,func);  

class= multi_SVM(pcaface,multiSVMstruct,npersons);
 
accuracy1=sum(class==RealClassTrain)/length(class);  

[testface0,realclass]=read_face(npersons,1);
testface = data_process(testface0,data_process_method);

testface2 = calculate_embedding('pca',testface,V,ma);
class= multi_SVM(testface2,multiSVMstruct,npersons);  

accuracy2=sum(class==realclass)/length(class);  
%msgbox(['ʶ��׼ȷ�ʣ�',num2str(accuracy*100),'%��'])
fprintf("ѵ��׼ȷ��:%.2f %% %.2f %%\n",accuracy1*100,accuracy2*100);

end