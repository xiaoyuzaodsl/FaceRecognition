path = './orl_faces/s1/1.pgm';
A=imread(path);
[cA,cH,cV,cD]=dwt2(A,'haar');%ʹ��haarС��
figure,imshow(A);title('ԭͼ');
figure,subplot(2,2,1),imshow(uint8(cA)),title('��Ƶ����');
subplot(2,2,2),imshow(uint8(cH)),title('ˮƽϸ�ڷ���');
subplot(2,2,3),imshow(uint8(cV)),title('��ֱϸ�ڷ���');
subplot(2,2,4),imshow(uint8(cD)),title('�Խ���ϸ�ڷ���');
fprintf("%d*%d\n",size(cA,1),size(cA,2));