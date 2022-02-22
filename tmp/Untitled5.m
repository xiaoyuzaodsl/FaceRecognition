npersons=40;%选取40个人的脸  
k=20;%降维至1维 
gamma=0.1; %svm参数
c=10;  %svm参数

global imgrow;  
global imgcol;  
imgrow=112;  
imgcol=92;  
data_process_method = 'mean';
[f0_matrix, RealClassTrain] =ReadFace(npersons,0);%读取训练数据
f_matrix = data_preprocess(f0_matrix,data_process_method);

nfaces=size(f_matrix,1);%样本人脸的数量  
ma = mean(f_matrix);
[pcaface,V]=myPCA2(f_matrix,k,ma);%主成分分析法特征提取  
   
multiSVMstruct=svm_train(pcaface,npersons,gamma,c);  

class= multi_SVM(pcaface,multiSVMstruct,npersons);
rclass = zeros(size(class,1),1);
for i = 1:size(class,1)
    rclass(i) = RealClassTrain(class(i));
end
accuracy1=sum(class==RealClassTrain)/length(class);  
%msgbox(['训练准确率：',num2str(accuracy*100),'%。'])
%fprintf("训练准确率:%d%%\n",accuracy*100);
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
%msgbox(['识别准确率：',num2str(accuracy*100),'%。'])
fprintf("训练准确率:%.2f %% %.2f %%\n",accuracy1*100,accuracy2*100);