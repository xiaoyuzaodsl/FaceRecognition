function [ scaledface] = scaling( faceMat,lowvec,upvec )  
%�������ݹ淶��  
%���롪��faceMat��Ҫ���й淶����ͼ�����ݣ�  
%       lowvecԭ������Сֵ  
%       upvecԭ�������ֵ  

%�����ݹ�һ����-1��1֮��
upnew=1;  
lownew=-1;  
[m,n]=size(faceMat);  
scaledface=zeros(m,n);  
for i=1:m  
    scaledface(i,:)=lownew+(faceMat(i,:)-lowvec)./(upvec-lowvec)*(upnew-lownew);  
end  
end  

