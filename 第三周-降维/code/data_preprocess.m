function [output] = data_preprocess(data,method)
if strcmp(method,'raw')
    output = data;
elseif strcmp(method,'trivial')
    output = double(data) / double(255.0);
elseif strcmp(method,'mean')
%     [m,n] = size(data);
%     output = zeros(m,n);
%     for i = 1:m
%         data_sum = sum(data(i,:));
%         output(i,:) = double(data(i,:)) / double(data_sum);
%     end

      [m,n] = size(data);
      %data_sum = sum(sum(data)) / m;
      output = zeros(m,n);
      for i = 1:m
            data_sum = sum(sum(data(i,:)));
            output(i,:) = double(data(i,:)) / double(data_sum);
      end
elseif strcmp(method,'lbp')
    [m,n] = size(data);        
    row = 112;
    col = 92;
    output = zeros(m,row*col);
    for i = 1:m
        tmp_data = reshape(data(i,:),[row,col]);
        tmp_output = convolution(tmp_data);
        output(i,:) = tmp_output(:)';
    end
    output = cal_tot_img(output);
    %��ʱoutput�ǰ���15*15����ָ��uniform lbpֱ��ͼ����area_num*59
    %output = output / double(6*7*59);
    %output = output / double(16*13*59);
end
end

function [output] = convolution(input)
%����Ϊ�˷������ʹ�õ���0��չ��ľ�����
[m0,n0] = size(input);
data = zeros(m0+2,n0+2);
data(2:m0+1,2:n0+1) = input(1:m0,1:n0);
[m,n] = size(data);
output = zeros(m-2,n-2);
    for i = 2:m-1
        for j = 2:n-1
            %%����lbp�ķ������൱��һ��3*3�ľ����ɨ��ȫͼ
            center = data(i,j);
            output(i-1,j-1) = output(i-1,j-1) + (data(i-1,j-1) > center) * pow2(7);
            output(i-1,j-1) = output(i-1,j-1) + (data(i,j-1) > center) * pow2(6);
            output(i-1,j-1) = output(i-1,j-1) + (data(i+1,j-1) > center) * pow2(5);
            output(i-1,j-1) = output(i-1,j-1) + (data(i-1,j) > center) * pow2(4);
            output(i-1,j-1) = output(i-1,j-1) + (data(i+1,j) > center) * pow2(3);
            output(i-1,j-1) = output(i-1,j-1) + (data(i-1,j+1) > center) * pow2(2);
            output(i-1,j-1) = output(i-1,j-1) + (data(i,j+1) > center) * pow2(1);
            output(i-1,j-1) = output(i-1,j-1) + (data(i+1,j+1) > center) * pow2(0);
        end
    end
end

function [output] = cal_tot_img(input)
[m,n] = size(input);
output = zeros(m,7*6*59);
%output = zeros(m,16*13*59);
for i = 1:m
    output(i,:) = cal_single_img(input(i,:));
end
end

function [output] = cal_single_img(input)
%�������������һ�е�
row = 112; col = 92;
input = reshape(input,[row,col]);
%����ָ�Ϊ15*15������
%�������о�̫���ˣ��޸�Ϊ5*5
area_d = 15;
%area_d = 7;
row_ = floor(row/area_d);col_ = floor(col/area_d);%�ܼ�row_*col_������
%fprintf("row = %d and col = %d\n",row_,col_);
x0 = 1;y0 = 1;%���½�
output = zeros(row_*col_,59);
id = 1;
for i = 1:row_
    for j = 1:col_
        c_x = x0 + (i - 1)* area_d;c_y = y0 + (j - 1)* area_d;
        area = input(c_x:c_x+area_d-1,c_y:c_y+area_d-1);
        output(id,:) = reshape(histogram(area),[1,59]);
        id = id + 1;
    end
end
%��ʱoutput�����[row*col,256]�ľ���ÿһ�д�����һ������ĻҶ�ֱ��ͼ
%�������ֱ�ӽ���ʹ�õĻ�ÿ������256ά�ᵼ�´����ÿ��ͼ����row*col*256 = 7*6*256=10752��С
%������ʹ��lda�ķ�����ά�м��sw��sb����ֱ��10752*10752��Ч�������Ż�һ��
%�����ܳ��������������⣬������Ҫ��output����һ��uniform lbp�Ĳ���(��������˼���ֱ��ͼ�ڲ��ˣ�����outputֻ��59��)
output = reshape(output,1,[]);
end


function [out_] = histogram(input)
%��������������һ��������ͼƬ������
%��Ϊ��������3*3�������Բ���Ҫ��һ����
output = zeros(256,1);
for i = 1:256
    output(i) = sum(input(:)==(i-1));
end
tmp_excel = excel();
%fprintf("tmp_excel\n");disp((tmp_excel'));
out_ = zeros(59,1);
for i = 1:256
    tmp_id = tmp_excel(i);
    out_(tmp_id) = out_(tmp_id) + output(i);
end
%disp("histogram out_\n");disp(out_');
end

function [out_] = excel()
output = zeros(256,1);
for i = 1:256
    if exchange(i) == 3
        %0000101,�Ĵ�����
        output(i) = 5;
    else
        output(i) = i;
    end
end
out_ = zeros(256,1);
data = unique(output);
%fprintf("data\n");disp(data);
for i = 1:256
    [row,col] = find(data==output(i));
    out_(i) = row;
end
%fprintf("out_\n");disp(out_');
end

function [output_] = exchange(input)
%input��һ��8λ�����ƣ���β��Ӹ���01������Ŀ����ת��
output = zeros(8,1);
num = zeros(8,1);
data = mod(input,pow2(8));
for i = 1:8
    output(i) = mod(data,2);
    data = (data - output(i))/2;
end
for i = 1:8
    num(i) = output(9-i);
end
%��ʱnum��ʾ�����Ʊ�ʾ
output_ = 0;
for i = 1:7
    output_ = output_ + mod(num(i)+num(i+1),2);
end
output_ = output_ + mod(num(1)+num(8),2);
if output_> 2
    output_ = 3;
end
end