npersons=40;%ѡȡ40���˵���  
k=20;%��ά��1ά 
gamma=0.01; %svm����
c=1;  %svm����
kernal_func = 'gauss';


global imgrow;  
global imgcol;  
global edit2  
imgrow=112;  
imgcol=92;  
  
set(edit2,'string','��ȡѵ������......')%��ʾ�ھ��Ϊedit2���ı�����  
drawnow     %���´��ڵ����ݣ���Ȼ�������ʱ�Ż���ʾ������ֻ�ܿ������һ��  
[f_matrix, RealClassTrain] =read_face(npersons,0);%��ȡѵ������  
nfaces=size(f_matrix,1);%��������������  
  
set(edit2,'string','ѵ������PCA������ȡ......')  
drawnow  
mA=mean(f_matrix);  
[pcaface,V]=myPCA(f_matrix,k,mA);%���ɷַ�����������ȡ  
  
set(edit2,'string','ѵ�����ݹ淶��......')  
drawnow  
  
set(edit2,'string','SVM����ѵ���������ĵȴ�������������ť......')  
drawnow  
multiSVMstruct=multi_svm_struct(pcaface,npersons,gamma,c,kernal_func);  
%save('recognize.mat','multiSVMstruct','npersons','k','mA','V','lowvec','upvec');  

class= multi_SVM(pcaface,multiSVMstruct,npersons);  
accuracy=sum(class==RealClassTrain)/length(class);  
msgbox(['ѵ��׼ȷ�ʣ�',num2str(accuracy*100),'%��'])

  
set(edit2,'string','��ȡ��������......')  
drawnow  
[testface,realclass]=read_face(npersons,1);  
  
set(edit2,'string','��������������ά......')  
drawnow  
m=size(testface,1);  
for i=1:m  
    testface(i,:)=testface(i,:)-mA;  
end  
pcatestface=testface*V;  
  
set(edit2,'string','�������ݹ淶��......')  
drawnow  
  
set(edit2,'string','SVM��������......')  
drawnow  
class= multi_SVM(pcatestface,multiSVMstruct,npersons);  
set(edit2,'string','������ɣ�����ѡ����Ƭ����������ʶ��')  
accuracy=sum(class==realclass)/length(class);  
msgbox(['ʶ��׼ȷ�ʣ�',num2str(accuracy*100),'%��'])