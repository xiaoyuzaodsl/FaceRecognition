global img_row img_col data_path
img_row = 112;
img_col = 92;
data_path = 'C:\Users\likaixu\Desktop\orl_faces';
[face_matrix, real_class] = read_face(41,1);
%sprintf("face_matrix = %d\n",face_matrix)
%�����Ǵ�face_matrix�������¹���ͼƬ�۲��Ƿ����
[m,n] = size(face_matrix);
for i = 1:m
    data = face_matrix(i,:);
    data = reshape(data,img_row,img_col);
    my_name = sprintf("output\\%d.png",i);
    imwrite(uint8(data),char(my_name));
end

function [face_matrix,real_class] = read_face(num_person,flag)
%������
%����: num_person:���� flag:ѵ��/����
%���:face_matrix:num_person*(img_num/2)*(size)��С��ÿ�б�ʾһ������ͼƬ
     %real_class:����ͼƬ�ı�ţ���Ӧԭ���ݵ��ļ�����
%Ϊ�˷��㣬����ļ�������ٶ��������ļ�������ͼ������һ��
global data_path img_row img_col
[file_name_list,file_num] = file_list(data_path);
% for i = 1:file_num
%      fprintf("%s ",file_name_list(i));
% end
%fprintf("\n");
exp_name = sprintf("%s\\%s",data_path,file_name_list(1));
[none_name,tot_num] = file_list(char(exp_name));
train_num = uint8(tot_num/2);
%fprintf("%d",train_num);
face_matrix = zeros(num_person*train_num,img_row*img_col);
real_class = zeros(num_person*train_num,1);
for i = 1:num_person
    tmp_file_name = file_name_list(i);
    tmp_id = extractAfter(tmp_file_name,'s');
    id = uint8(str2num(char(tmp_id)));
    now_path = sprintf("%s\\%s",data_path,tmp_file_name);
    %fprintf("now_path = %s\n",char(now_path));
    if flag==0
        [data_matrix,id_matrix] = insert_img(char(now_path),id);%����ѵ��
    else
        [data_matrix,id_matrix] = insert_img_validation(char(now_path),id);%���ڲ���
    end
    %fprintf("now insert_img over");
    face_matrix(i*train_num-train_num+1:i*train_num,:) = data_matrix;
    real_class(i*train_num-train_num+1:i*5,:) = id_matrix;
end
end


function [file_name_list,file_num] = file_list(path)
%���pathĿ¼�����е��ļ������ļ�������������.��..��
raw_files = dir(path);
size0 = size(raw_files);
file_num = size0(1);
%fprintf("file_num = %d\n",file_num);
file_name_list = strings(1,file_num-2);
for i =3:file_num
    %fprintf("%s ",raw_files(i).name);
    file_name_list(i-2) = raw_files(i).name;
end
%fprintf("\n");
file_name_list = sort(file_name_list,1);
file_num = file_num - 2;%��Ҫȥ��.��..�����ļ���
end

function [data_matrix,id_matrix] = insert_img(path,id)
global img_row
global img_col
[img_name_list,img_num] = file_list(path);
used_img_num = uint8(img_num / 2);
data_matrix = zeros(used_img_num,img_row*img_col);
id_matrix = zeros(used_img_num,1);
%��ʱflag = 0����ʾ����ѵ��
for i = 1:used_img_num
    single_img_path = sprintf("%s\\%s",path,img_name_list(i));
    %fprintf("%s\n",single_img_path);
    single_img = imread(char(single_img_path));
    data_matrix(i,:) = single_img(:);
    id_matrix(i) = id;
end
end

function [data_matrix,id_matrix] = insert_img_validation(path,id)
global img_row
global img_col
[img_name_list,img_num] = file_list(path);
used_img_num = uint8(img_num / 2);
data_matrix = zeros(used_img_num,img_row*img_col);
id_matrix = zeros(used_img_num,1);
%��ʱflag = 0����ʾ����ѵ��
for i = 1:used_img_num
    single_img_path = sprintf("%s\\%s",path,img_name_list(i+used_img_num));
    %fprintf("%s\n",single_img_path);
    single_img = imread(char(single_img_path));
    data_matrix(i,:) = single_img(:);
    id_matrix(i) = id;
end
end