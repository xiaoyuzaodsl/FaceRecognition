data_process_method='mean';
npersons=40;%ѡȡ40���˵���  
k=20;%��ά��1ά 
global imgrow;  
global imgcol;  
imgrow=112;  
imgcol=92;  

[f_matrix, temp_id] =ReadFace(npersons,0);%��ȡѵ������
ma = mean(f_matrix);
[pcaface,V]=myPCA2(f_matrix,k,ma);%���ɷַ�����������ȡ  
class = onehot(temp_id);
img = pcaface';
label = class';
% rng('default');
% num_hid1 = 100;
% % ��Ϊ���Ա�������Ҳ�����޼ලѧϰ�㷨����˲���ҪĿ��ֵ train_y �Ĳ���
% ae1 = trainAutoencoder(img, num_hid1, ...
%     'MaxEpochs', 400, ...
%     'L2WeightRegularization', .004, ...
%     'SparsityRegularization', 4, ...
%     'SparsityProportion', 0.15, ...
%     'ScaleData', false);
% % ʹ�õ�һ���Ա������õ����Ӧ��ѹ�����룬
% feat1 = encode(ae1, img);
% num_hid2 = 50;
% ae2 = trainAutoencoder(feat1, num_hid2, ...
%         'MaxEpochs', 100, ...
%         'L2WeightRegularization', .002, ...
%         'SparsityRegularization', 4, ...
%         'SparsityProportion', 0.1, ...
%         'ScaleData', false);
% %view(ae2);
% % ʹ�õڶ����Ա������õ����Ӧ��ѹ������
% feat2 = encode(ae2, feat1);
% softnet = trainSoftmaxLayer(feat2, label, 'MaxEpochs', 400);
% deepnet = stack(ae1, ae2, softnet);
% %view(deepnet)
% deepnet = train(deepnet,img, label);
% res = deepnet(img);

[testnet]=deep_train(img,label);
testnet = train(testnet,img, label);
res = testnet(img);
output = output_guess(res);

[input2_test,label2_test] = ReadFace(npersons,1);
mypca_test = cal_latent('pca',input2_test,V,ma);
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
%�����������deepnet�Ľ����Ҫ�����Ԥ��
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