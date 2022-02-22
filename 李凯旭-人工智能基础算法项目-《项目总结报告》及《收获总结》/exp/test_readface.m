npersons = 41;
flag = 0;
global imgrow;  
global imgcol;  
imgrow = 112;
imgcol = 92;
[f_matrix,realclass] = read_face(npersons,flag);
for i = 1:size(f_matrix,1)
   img_line = f_matrix(i,:);
   img_class = realclass(i);
   img_output = reshape(img_line,[imgrow,imgcol]);
   img_id = i - floor((i-1)/5) * 5;
   img_name = sprintf('./output_exp1/%d_%d.png',img_class,img_id);
   %注意这里需要使用uint8才能正常输出，否则均为空白
   imwrite(uint8(img_output),img_name);
end