% [train_x, train_y] = digitTrainCellArrayData;
% inputSize = 28*28;
% xTest = zeros(inputSize,numel(train_x));
% for i = 1:numel(train_x)
%     xTest(:,i) =train_x{i}(:);
% end

[f, tl] =ReadFace(40,0);
class = onehot(tl);
ma = mean(f);
[xx,V]=myPCA2(f,20,ma);
xTest = xx';
train_y = class';

rng('default');
num_hid1 = 100;
% ��Ϊ���Ա�������Ҳ�����޼ලѧϰ�㷨����˲���ҪĿ��ֵ train_y �Ĳ���
ae1 = trainAutoencoder(xTest, num_hid1, ...
    'MaxEpochs', 200, ...
    'L2WeightRegularization', .004, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', 0.15, ...
    'ScaleData', false);
% ʹ�õ�һ���Ա������õ����Ӧ��ѹ�����룬
feat1 = encode(ae1, xTest);
num_hid2 = 50;
ae2 = trainAutoencoder(feat1, num_hid2, ...
        'MaxEpochs', 100, ...
        'L2WeightRegularization', .002, ...
        'SparsityRegularization', 4, ...
        'SparsityProportion', 0.1, ...
        'ScaleData', false);
view(ae2);
% ʹ�õڶ����Ա������õ����Ӧ��ѹ������
feat2 = encode(ae2, feat1);
softnet = trainSoftmaxLayer(feat2, train_y, 'MaxEpochs', 200);
deepnet = stack(ae1, ae2, softnet);
view(deepnet)
deepnet = train(deepnet,xTest, train_y);
res = deepnet(xTest);


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