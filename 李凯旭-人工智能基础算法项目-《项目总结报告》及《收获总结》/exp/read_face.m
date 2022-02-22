function [f_matrix,realclass] = read_face(npersons,flag)
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
imgrow = 112;
imgcol = 92;
realclass=zeros(npersons*5,1);  
f_matrix=zeros(npersons*5,imgrow*imgcol);
path_prefix = './orl_faces/s';
for i=1:npersons     
    face_path = sprintf('%s%d/',path_prefix,i);
    for j=1:5  
        if flag==0 %读入训练样本图像的数据  
            img_name = sprintf('%s%d.pgm',face_path,j);
            realclass((i-1)*5+j)=i;
        else %读入测试样本数据  
            img_name = sprintf('%s%d.pgm',face_path,j+5);
            realclass((i-1)*5+j)=i;  
        end  
        img=imread(img_name);  
        f_matrix((i-1)*5+j,:)=img(:)';  
    end  
end  
end  