function [sw,sb,jw] = lda2(data,label,k)
[m,n] = size(data);
fprintf("m = %d and n = %d\n",m,n);
u = mean(data);
%我们这里知道我们的数据都是5行一组
class_num = uint8(m/5);
fprintf("class_num = %d\n",class_num);
u_i = zeros(class_num,n);
sw = zeros(n,n);
sb = zeros(n,n);
for i = 1:class_num
    class_data = data(i*5-4:i*5,:);
    u_i(i,:) = mean(class_data);
end

for i = 1:m
    tag = floor((i-1)/5+1);
    sw_step = data(i,:)-u_i(tag,:);
    sw = sw + (sw_step)'*(sw_step);
end

for i = 1:class_num
    sb_step = u_i(i,:)-u;
   sb = sb + 5*(sb_step)'*(sb_step); 
end

%sb(isnan(sb)) = 0; sw(isnan(sw)) = 0;
%sb(isinf(sb)) = 0; sw(isinf(sw)) = 0;


jw = inv(sw)*sb;
% fprintf("sw\n");disp(sw);fprintf("\n");
% fprintf("sb\n");disp(sb);fprintf("\n");
% fprintf("sw^-1");disp(inv(sw));fprintf("\n");
% fprintf("jw\n");disp(jw);fprintf("\n");
%jw(isinf(jw)) = 0; jw(isinf(jw)) = 0;
%[W,D] = eigs(jw,k);
%lda = data*W;
end