path = './orl_faces/s1/1.pgm';
A=imread(path);
[cA,cH,cV,cD]=dwt2(A,'haar');%使用haar小波
figure,imshow(A);title('原图');
figure,subplot(2,2,1),imshow(uint8(cA)),title('低频分量');
subplot(2,2,2),imshow(uint8(cH)),title('水平细节分量');
subplot(2,2,3),imshow(uint8(cV)),title('垂直细节分量');
subplot(2,2,4),imshow(uint8(cD)),title('对角线细节分量');
fprintf("%d*%d\n",size(cA,1),size(cA,2));