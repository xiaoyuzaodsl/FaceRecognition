data_process_method='lbp';
npersons=25;%选取40个人的脸  
k=10;%降维至1维 
global imgrow;  
global imgcol;  
imgrow=112;  
imgcol=92;  

[f_matrix, temp_id] =ReadFace(npersons,0);%读取训练数据
[f_matrix] = data_preprocess(f_matrix,data_process_method);
ma = mean(f_matrix);
[D,mylda,W]=lda(f_matrix,temp_id,k);
class = onehot(temp_id);
img = mylda';
label = class';

[testnet]=deep_train(img,label);
testnet = train(testnet,img, label);
res = testnet(img);
output = output_guess(res);

[input2_test,label2_test] = ReadFace(npersons,1);
[input2_test] = data_preprocess(input2_test,data_process_method);
mypca_test = cal_latent('lda',input2_test,W);
img2 = mypca_test';
res2 = testnet(img2);
output2 = output_guess(res2);
output2tmp = output2';
[m,n] = size(output2tmp);
guess_test = zeros(m,1);
for i=1:m
    [row,col] = find(1==output2tmp(i,:));
    guess_test(i) = col;
end

compared =  zeros(m,2);

for i=1:m
    compared(i,1) = label2_test(i);
    compared(i,2) = guess_test(i);
end
acc = sum(label2_test==guess_test) / length(guess_test);
fprintf("acc = %.2f%%\n",acc*100);

function [output]=onehot(label)
[m,n] = size(label);
output = zeros(m,40);
for i=1:m
    tmp = zeros(1,40);
    tmpx = label(i);
    tmp(1,tmpx) = 1;
    output(i,:) = tmp;
end
end

function [out] = output_guess(poss)
%这里输入的是deepnet的结果，要求输出预测
[m,n] = size(poss);
out = zeros(m,n);
for i = 1:n
    tmp = poss(:,i);
    value = max(max(tmp));
    [row,col] = find(value==tmp);
    x = min(row);
    out(x,i) = 1;
end
end


function [output] = cal_latent(func_name,test_data,V,mA)
if strcmp(func_name,'pca')
    output = (test_data-mA)*V;
elseif strcmp(func_name,'lda')
    output = test_data*V;
end

end