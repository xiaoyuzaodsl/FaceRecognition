npersons=40;%选取40个人的脸  
k=20;%降维至1维 
gamma=0.01; %svm参数
c=1;  %svm参数
kernal_func = 'gauss';


global imgrow;  
global imgcol;  
global edit2  
imgrow=112;  
imgcol=92;  
  
set(edit2,'string','读取训练数据......')%显示在句柄为edit2的文本框里  
drawnow     %更新窗口的内容，不然程序结束时才会显示，这样只能看到最后一句  
[f_matrix, RealClassTrain] =read_face(npersons,0);%读取训练数据  
nfaces=size(f_matrix,1);%样本人脸的数量  
  
set(edit2,'string','训练数据PCA特征提取......')  
drawnow  
mA=mean(f_matrix);  
[pcaface,V]=myPCA(f_matrix,k,mA);%主成分分析法特征提取  
  
set(edit2,'string','训练数据规范化......')  
drawnow  
  
set(edit2,'string','SVM样本训练，请耐心等待，勿点击其他按钮......')  
drawnow  
multiSVMstruct=multi_svm_struct(pcaface,npersons,gamma,c,kernal_func);  
%save('recognize.mat','multiSVMstruct','npersons','k','mA','V','lowvec','upvec');  

class= multi_SVM(pcaface,multiSVMstruct,npersons);  
accuracy=sum(class==RealClassTrain)/length(class);  
msgbox(['训练准确率：',num2str(accuracy*100),'%。'])

  
set(edit2,'string','读取测试数据......')  
drawnow  
[testface,realclass]=read_face(npersons,1);  
  
set(edit2,'string','测试数据特征降维......')  
drawnow  
m=size(testface,1);  
for i=1:m  
    testface(i,:)=testface(i,:)-mA;  
end  
pcatestface=testface*V;  
  
set(edit2,'string','测试数据规范化......')  
drawnow  
  
set(edit2,'string','SVM样本分类......')  
drawnow  
class= multi_SVM(pcatestface,multiSVMstruct,npersons);  
set(edit2,'string','测试完成！可以选择照片，进行人脸识别！')  
accuracy=sum(class==realclass)/length(class);  
msgbox(['识别准确率：',num2str(accuracy*100),'%。'])