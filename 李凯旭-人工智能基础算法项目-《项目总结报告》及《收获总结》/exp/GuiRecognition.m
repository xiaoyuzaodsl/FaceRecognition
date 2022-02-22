global h_axes1  
global h_axes2  
global edit2  
npersons = 40;
k = 20;
func = 'gauss';
gamma = 0.01;
c = 1;
data_process_method = 'raw';
[f0_matrix, RealClassTrain] =read_face(npersons,0);%读取训练数据
f_matrix = data_process(f0_matrix,data_process_method);

ma = mean(f_matrix);
[pcaface,V]=myPCA(f_matrix,k,ma);%主成分分析法特征提取  
   
multiSVMstruct=multi_svm_struct(pcaface,npersons,gamma,c,func);

img=getimage(h_axes1);%获得之前选中的照片的信息  
if isempty(img)  
    msgbox('请先选择一张图片！')  
    %break  
end  
testface=img(:)';  
set(edit2,'string','测试数据降维......')  
drawnow  
disp('测试数据特征降维...')  
disp('.................................................')  
Z=double(testface)-ma;  
pcatestface=Z*V;  
set(edit2,'string','测试特征数据规范化......')  
drawnow  
disp('SVM样本识别...')  
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
set(edit2,'string','识别完成！')  
drawnow  
axes(h_axes2);  
imshow(imread(['.\orl_faces\s',num2str(class),'\1.pgm']));  
msgbox(['样本识别为第',num2str(class),'个人'])