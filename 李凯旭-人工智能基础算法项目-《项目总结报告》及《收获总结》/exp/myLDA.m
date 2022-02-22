function [D,lda,W] = myLDA(data,label,k)
[m,n] = size(data);
u = mean(data);
class_num = uint8(m/5);
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

sb(isnan(sb)) = 0; sw(isnan(sw)) = 0;
sb(isinf(sb)) = 0; sw(isinf(sw)) = 0;

jw = inv(sw)*sb;
jw(isinf(jw)) = 0; jw(isinf(jw)) = 0;
[W,D] = eigs(jw,k);
lda = data*W;
end