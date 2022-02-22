function [f_matrix,realclass] = ReadFace(npersons,flag)
%读取ORL人脸库照片里的数据到矩阵  
%输入：  
%     npersons-需要读入的人数,每个人的前五幅图为训练样本，后五幅为验证样本  
%     imgrow-图像的行像素为全局变量  
%     imgcol-图像的列像素为全局变量  
%     flag-标志，为0表示读入训练样本，为1表示读入测试样本  
%输出：  
%     f_matrix-一行是一张照片，列数为112*92=10304。行数为npersons*10
%     realclass-行数为npersons*10，列数为1。记录是第几个人。
%注意：
%已知全局变量：imgrow=112; imgcol=92;  
global imgrow;  
global imgcol;   
realclass=zeros(npersons*5,1);  
f_matrix=zeros(npersons*5,imgrow*imgcol);  
for i=1:npersons  
    facepath='C:\Users\likaixu\Desktop\orl_faces\s';  
    facepath=strcat(facepath,num2str(i));  
    facepath=strcat(facepath,'\');  
    cachepath=facepath;  
    for j=1:5  
        facepath=cachepath;  
        if flag==0 %读入训练样本图像的数据  
            facepath=strcat(facepath,'0'+j);  
            realclass((i-1)*5+j)=i;
        else %读入测试样本数据  
            facepath=strcat(facepath,num2str(5+j));  
            realclass((i-1)*5+j)=i;  
        end  
        facepath=strcat(facepath,'.pgm');  
        img=imread(facepath);  
        f_matrix((i-1)*5+j,:)=img(:)';  
    end  
end  
end  
