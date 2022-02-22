num_person = 41;
flag = 0;
[data,label] = read_face(num_person,flag);
[output] = data_process(data,'lbp');
path = './output_LBPimg';
[m,n] = size(output);
global imgrow
global imgcol
id = 1;
for i = 1:m
    tmp_data = output(i,:);
    tmp_data = reshape(tmp_data,[imgrow,imgcol]);
    my_name = sprintf("%s/%d_%d.png",path,label(i),i - floor((i-1)/5)*5);
    imwrite(uint8(tmp_data),char(my_name));
end