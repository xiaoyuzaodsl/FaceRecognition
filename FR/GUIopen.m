global h_axes1  
[filename,pathname]=uigetfile({'*.pgm';'*.jpg';'*.tif';'*.*'},'��ѡ��һ������ʶ�����Ƭ');  
if filename==0  
    msgbox('��ѡ��һ����Ƭ�ļ�')  
else  
    filepath=[pathname,filename];  
    axes(h_axes1);  
    imshow(imread(filepath));  
end  