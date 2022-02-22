k = 2;
data = Iris_tree_preprocess();
%下面来分配数据，已知1-50为1类，51-100为2类，101-150为三类
%所以每类的前25个归为训练，后25个归为测试
train_f = zeros(75,4);train_l = zeros(75,1);
test_f = zeros(75,4);test_l = zeros(75,1);
for i =1:25
    train_f(i,:) = data(i,1:4);
    train_l(i) = data(i,5);
    test_f(i,:) = data(i+25,1:4);
    test_l(i) = data(i+25,5);
end
for i =51:75
    train_f(i-25,:) = data(i,1:4);
    train_l(i-25) = data(i,5);
    test_f(i-25,:) = data(i+25,1:4);
    test_l(i-25) = data(i+25,5);
end
for i =101:125
    train_f(i-50,:) = data(i,1:4);
    train_l(i-50) = data(i,5);
    
    test_f(i-50,:) = data(i+25,1:4);
    test_l(i-50) = data(i+25,5);
end

[D,mylda,W] = lda(train_f,train_l,k);% 
mylda_test = cal_latent('lda',test_f,W);
[guess_label,test_acc2] = test_accuracy_(mylda_test,test_l,mylda,train_l);
fprintf("LDA acc2 = %f\n",test_acc2);

% [mylda,mapping] = FisherLDA(train_f,train_l,k); 
% mylda_test = cal_latent('lda',test_f,mapping.M);
% [guess_label,test_acc2] = test_accuracy_(mylda_test,test_l,mylda,train_l);
% fprintf("LDA acc2 = %f\n",test_acc2);

% [pcaF,eigV,mF] = myPCA(train_f,k);
% mypca_test = cal_latent('pca',test_f,eigV,mF);
% [m,n] = size(pcaF);
% [guess_label,test_acc] = test_accuracy_(mypca_test,test_l,pcaF,train_l);
% fprintf("PCA acc = %f\n",test_acc);



function [output] = cal_latent(func_name,test_data,V,mA)
if strcmp(func_name,'pca')
    output = (test_data-mA)*V;
elseif strcmp(func_name,'lda')
    output = test_data*V;
end

end

function [mF,label,eigF,eigV] = myeigen_face(num_person,flag,k)
global img_row img_col
  [orF,label] = read_face(num_person,flag);
  [pcaF,eigV,mF] = myPCA(orF,k);
  [m,n] = size(orF);
  eigF = pcaF;
end

function [guess_label,acc] = test_accuracy_(test_img,test_label,train_img,train_label)
[m,n] = size(test_img); fprintf("test_img m = %d and n = %d\n",m,n);
[m2,n2] = size(train_img); fprintf("train_img m = %d and n = %d\n",m2,n2);
guess_label = zeros(m,1);
for i = 1:m
    guess_label(i) = cal_label(test_img(i,:),train_img,train_label);
end
delta_test = guess_label - test_label;
acc_num = sum(delta_test(:)==0);
acc = double(acc_num) / double(m);
end

function [output] = cal_label(img,dataset,label)
    [m,n] = size(dataset);
    dis_list = zeros(m,1);
    for i=1:m
        dis_list(i) = dis_L2(img,dataset(i,:));
    end
    [row,col] = find(dis_list==min(min(dis_list)));
    output = label(row);
end
    
function [dis] = dis_L1(img1,img2)
delta_dis = img1 - img2;
dis = sum(abs(delta_dis(:)));
end

function [dis] = dis_L2(img1,img2)
delta_dis = img1 - img2;
dis_abs = abs(delta_dis(:));
[m,~] = size(dis_abs);
dis = 0;
for i = 1:m
    dis = dis + dis_abs(i) * dis_abs(i);
end
end