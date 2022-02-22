num_person = 41;
[data,label_test] = read_face(num_person,0);
[output] = data_preprocess(data,'lbp');
path = 'C:\Users\likaixu\Desktop\output3';
[m,n] = size(output);
row = 110;
col = 90;
id = 1;
for i = 1:m
    tmp_data = output(i,:);
    tmp_data = reshape(tmp_data,row,col);
    my_name = sprintf("output3\\%d.png",id);
    id = id + 1;
    imwrite(uint8(tmp_data),char(my_name));
end