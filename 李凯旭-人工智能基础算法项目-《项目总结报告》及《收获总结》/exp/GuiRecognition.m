global h_axes1  
global h_axes2  
global edit2  
npersons = 40;
k = 20;
func = 'gauss';
gamma = 0.01;
c = 1;
data_process_method = 'raw';
[f0_matrix, RealClassTrain] =read_face(npersons,0);%��ȡѵ������
f_matrix = data_process(f0_matrix,data_process_method);

ma = mean(f_matrix);
[pcaface,V]=myPCA(f_matrix,k,ma);%���ɷַ�����������ȡ  
   
multiSVMstruct=multi_svm_struct(pcaface,npersons,gamma,c,func);

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
Z=double(testface)-ma;  
pcatestface=Z*V;  
set(edit2,'string','�����������ݹ淶��......')  
drawnow  
disp('SVM����ʶ��...')  
disp('.................................................')  
voting=zeros(1,npersons);  
for i=1:npersons-1  
    for j=i+1:npersons  
        class=svmclassify(multiSVMstruct{i}{j},pcatestface);  
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