function [f_matrix,realclass] = read_face(npersons,flag)
%��ȡORL��������Ƭ������ݵ�����  
%���룺  
%     npersons-��Ҫ���������,ÿ���˵�ǰ���ͼΪѵ�������������Ϊ��֤����  
%     imgrow-ͼ���������Ϊȫ�ֱ���  
%     imgcol-ͼ���������Ϊȫ�ֱ���  
%     flag-��־��Ϊ0��ʾ����ѵ��������Ϊ1��ʾ�����������  
%�����  
%     f_matrix-һ����һ����Ƭ������Ϊ112*92=10304������Ϊnpersons*10
%     realclass-����Ϊnpersons*10������Ϊ1����¼�ǵڼ����ˡ�
%ע�⣺
%��֪ȫ�ֱ�����imgrow=112; imgcol=92;  
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
        if flag==0 %����ѵ������ͼ�������  
            img_name = sprintf('%s%d.pgm',face_path,j);
            realclass((i-1)*5+j)=i;
        else %���������������  
            img_name = sprintf('%s%d.pgm',face_path,j+5);
            realclass((i-1)*5+j)=i;  
        end  
        img=imread(img_name);  
        f_matrix((i-1)*5+j,:)=img(:)';  
    end  
end  
end  