function [f_matrix,realclass] = ReadFace(npersons,flag)
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
realclass=zeros(npersons*5,1);  
f_matrix=zeros(npersons*5,imgrow*imgcol);  
for i=1:npersons  
    facepath='C:\Users\likaixu\Desktop\orl_faces\s';  
    facepath=strcat(facepath,num2str(i));  
    facepath=strcat(facepath,'\');  
    cachepath=facepath;  
    for j=1:5  
        facepath=cachepath;  
        if flag==0 %����ѵ������ͼ�������  
            facepath=strcat(facepath,'0'+j);  
            realclass((i-1)*5+j)=i;
        else %���������������  
            facepath=strcat(facepath,num2str(5+j));  
            realclass((i-1)*5+j)=i;  
        end  
        facepath=strcat(facepath,'.pgm');  
        img=imread(facepath);  
        f_matrix((i-1)*5+j,:)=img(:)';  
    end  
end  
end  
