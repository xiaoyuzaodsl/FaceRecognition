global h_axes1  
global h_axes2  
global edit2  
load('recognize.mat');  
set(edit2,'string','��ȡ��������......')  
drawnow  
disp('��ȡ��������...')  
disp('.................................................')  
img=getimage(h_axes1);%���֮ǰѡ�е���Ƭ����Ϣ  
if isempty(img)  
    msgbox('����ѡ��һ��ͼƬ��')  
    %break  
end  
testface=img(:)';  
set(edit2,'string','�������ݽ�ά......')  
drawnow  
disp('��������������ά...')  
disp('.................................................')  
Z=double(testface)-mA;  
pcatestface=Z*V;  
set(edit2,'string','�����������ݹ淶��......')  
drawnow  
disp('�����������ݹ淶��...')  
disp('.................................................')  
scaledtestface=-1+(pcatestface-lowvec)./(upvec-lowvec)*2;  
set(edit2,'string','SVM����ʶ��......')  
drawnow  
disp('SVM����ʶ��...')  
disp('.................................................')  
voting=zeros(1,npersons);  
for i=1:npersons-1  
    for j=i+1:npersons  
        class=svmclassify(multiSVMstruct{i}{j},scaledtestface);  
        voting(i)=voting(i)+(class==1);  
        voting(j)=voting(j)+(class==0);  
    end  
end  
[~,class]=max(voting);  
set(edit2,'string','ʶ����ɣ�')  
drawnow  
axes(h_axes2);  
imshow(imread(['.\orl_faces\s',num2str(class),'\1.pgm']));  
msgbox(['����ʶ��Ϊ��',num2str(class),'����'])