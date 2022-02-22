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
    %此时output是按照15*15区域分割的uniform lbp直方图，共area_num*59
    %output = output / double(6*7*59);
    %output = output / double(16*13*59);
end
end

function [output] = convolution(input)
%这里为了方便起见使用的是0扩展后的卷积结果
[m0,n0] = size(input);
data = zeros(m0+2,n0+2);
data(2:m0+1,2:n0+1) = input(1:m0,1:n0);
[m,n] = size(data);
output = zeros(m-2,n-2);
    for i = 2:m-1
        for j = 2:n-1
            %%按照lbp的方法，相当于一个3*3的卷积核扫描全图
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
%这里假设输入是一行的
row = 112; col = 92;
input = reshape(input,[row,col]);
%这里分割为15*15的区域
%修正，感觉太少了，修改为5*5
area_d = 15;
%area_d = 7;
row_ = floor(row/area_d);col_ = floor(col/area_d);%总计row_*col_个区域
%fprintf("row = %d and col = %d\n",row_,col_);
x0 = 1;y0 = 1;%左下角
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
%此时output是这个[row*col,256]的矩阵，每一行代表了一个区域的灰度直方图
%但是这个直接进行使用的话每个区域256维会导致处理后每个图都是row*col*256 = 7*6*256=10752大小
%最后如果使用lda的方法降维中间的sw，sb矩阵直接10752*10752，效果跟无优化一样
%还可能出现奇异矩阵的问题，所以需要对output进行一个uniform lbp的操作(这里放在了计算直方图内部了，所以output只有59行)
output = reshape(output,1,[]);
end


function [out_] = histogram(input)
%这里假设输入的是一个正常的图片（区域）
%因为本来就是3*3区域所以不需要归一化了
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
        %0000101,四次跳变
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
%input是一个8位二进制，首尾相接根据01跳变数目进行转化
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
%此时num表示二进制表示
output_ = 0;
for i = 1:7
    output_ = output_ + mod(num(i)+num(i+1),2);
end
output_ = output_ + mod(num(1)+num(8),2);
if output_> 2
    output_ = 3;
end
end