npersons=40;%ѡȡ40���˵���  
k=20;%��ά��1ά 
gamma=1; %svm����
c=1;  %svm����



global imgrow;  
global imgcol;  
global edit2  
imgrow=112;  
imgcol=92;  
  
set(edit2,'string','��ȡѵ������......')%��ʾ�ھ��Ϊedit2���ı�����  
drawnow     %���´��ڵ����ݣ���Ȼ�������ʱ�Ż���ʾ������ֻ�ܿ������һ��  
[f_matrix, RealClassTrain] =ReadFace(npersons,0);%��ȡѵ������  
nfaces=size(f_matrix,1);%��������������  
  
set(edit2,'string','ѵ������PCA������ȡ......')  
drawnow  
mA=mean(f_matrix);  
[pcaface,V]=fastPCA(f_matrix,k,mA);%���ɷַ�����������ȡ  
  
set(edit2,'string','ѵ�����ݹ淶��......')  
drawnow  
lowvec=min(pcaface);  
upvec=max(pcaface);  
scaledface = scaling( pcaface,lowvec,upvec);  
  
set(edit2,'string','SVM����ѵ���������ĵȴ�������������ť......')  
drawnow  
multiSVMstruct=multiSVMtrain( scaledface,npersons,gamma,c);  
save('recognize.mat','multiSVMstruct','npersons','k','mA','V','lowvec','upvec');  


class= multiSVM(scaledface,multiSVMstruct,npersons);  
accuracy=sum(class==RealClassTrain)/length(class);  
msgbox(['ѵ��׼ȷ�ʣ�',num2str(accuracy*100),'%��'])

  
set(edit2,'string','��ȡ��������......')  
drawnow  
[testface,realclass]=ReadFace(npersons,1);  
  
set(edit2,'string','��������������ά......')  
drawnow  
m=size(testface,1);  
for i=1:m  
    testface(i,:)=testface(i,:)-mA;  
end  
pcatestface=testface*V;  
  
set(edit2,'string','�������ݹ淶��......')  
drawnow  
scaledtestface = scaling( pcatestface,lowvec,upvec);  
  
set(edit2,'string','SVM��������......')  
drawnow  
class= multiSVM(scaledtestface,multiSVMstruct,npersons);  
set(edit2,'string','������ɣ�����ѡ����Ƭ����������ʶ��')  
accuracy=sum(class==realclass)/length(class);  
msgbox(['ʶ��׼ȷ�ʣ�',num2str(accuracy*100),'%��'])