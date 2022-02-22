% data_process_method='raw';
% data_process_method='lbp';
%data_process_method='uni_lbp';
data_process_method='wavelet';
npersons=40;%选取40个人的脸  
k=20;%降维至1维 
global imgrow;  
global imgcol;  
imgrow=112;  
imgcol=92;  

[f_matrix, temp_id] =read_face(npersons,0);%读取训练数据
[f_matrix] = data_process(f_matrix,data_process_method);

ma = mean(f_matrix);
[pcaface,V]=myPCA(f_matrix,k,ma);%主成分分析法特征提取  
class = onehot(temp_id);
img = pcaface';
label = class';
[testnet]=deep_ae_classify_unilbp(img,label);
testnet = train(testnet,img, label);
res = testnet(img);
output = output_guess(res);

[input2_test,label2_test] = read_face(npersons,1);
[input2_test] = data_process(input2_test,data_process_method);

mypca_test = calculate_embedding('pca',input2_test,V,ma);
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

function [deepnet]=deep_ae_classify_unilbp(img,label)
%注意，这里的img和label必须是保证一列一个的
%为了能够计算，这里的label需要修改为one-hot编码格式
rng('default');
num_hid1 = 50;
% 因为是自编码器，也属于无监督学习算法，因此不需要目标值 train_y 的参与
ae1 = trainAutoencoder(img, num_hid1, ...
    'MaxEpochs', 400, ...
    'L2WeightRegularization', .004, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', 0.15, ...
    'ScaleData', false);
% 使用第一个自编码器得到其对应的压缩编码，
feat1 = encode(ae1, img);
num_hid2 = 20;
ae2 = trainAutoencoder(feat1, num_hid2, ...
        'MaxEpochs', 400, ...
        'L2WeightRegularization', .002, ...
        'SparsityRegularization', 4, ...
        'SparsityProportion', 0.1, ...
        'ScaleData', false);
%view(ae2);
% 使用第二个自编码器得到其对应的压缩编码
feat2 = encode(ae2, feat1);
softnet = trainSoftmaxLayer(feat2, label, 'MaxEpochs', 400);
deepnet = stack(ae1, ae2, softnet);
end