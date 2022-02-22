npersons=40;%ѡȡ40���˵���  
k=20;%��ά��1ά 
gamma=0.1; %svm����
c=10;  %svm����

global imgrow;  
global imgcol;  
imgrow=112;  
imgcol=92;  
data_process_method = 'mean';
[f0_matrix, RealClassTrain] =ReadFace(npersons,0);%��ȡѵ������
f_matrix = data_preprocess(f0_matrix,data_process_method);

nfaces=size(f_matrix,1);%��������������  
ma = mean(f_matrix);
[pcaface,V]=myPCA2(f_matrix,k,ma);%���ɷַ�����������ȡ  
   
multiSVMstruct=svm_train(pcaface,npersons,gamma,c);  

class= multi_SVM(pcaface,multiSVMstruct,npersons);
rclass = zeros(size(class,1),1);
for i = 1:size(class,1)
    rclass(i) = RealClassTrain(class(i));
end
accuracy1=sum(class==RealClassTrain)/length(class);  
%msgbox(['ѵ��׼ȷ�ʣ�',num2str(accuracy*100),'%��'])
%fprintf("ѵ��׼ȷ��:%d%%\n",accuracy*100);
tmp_class = zeros(size(class,1),2);
for i = 1:size(tmp_class,1)
    tmp_class(i,1) = RealClassTrain(i,1);
    tmp_class(i,2) = rclass(i,1);
end
[testface0,realclass]=ReadFace(npersons,1);
testface = data_preprocess(testface0,data_process_method);
[testface2,V2] = myPCA2(testface,k,ma);

class= multi_SVM(testface2,multiSVMstruct,npersons);  

accuracy2=sum(class==realclass)/length(class);  
%msgbox(['ʶ��׼ȷ�ʣ�',num2str(accuracy*100),'%��'])
fprintf("ѵ��׼ȷ��:%.2f %% %.2f %%\n",accuracy1*100,accuracy2*100);