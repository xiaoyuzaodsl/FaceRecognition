k = 20;
npersons = 41;
data_process_method = 'raw';
% data_process_method = 'lbp';
data_process_method = 'norm_all';
%data_m2 = 'norm_all';
%数据读入
[input_train,label_train] = read_face(npersons,0);
[input_test,label_test] = read_face(npersons,1);
input_train_ = data_process(input_train,data_process_method);
input_test_ = data_process(input_test,data_process_method);
%input_train_ = data_process(input_train_,data_m2);
%input_test_ = data_process(input_test_,data_m2);
mF = mean(input_train_,1);
%PCA准确性
% mF = mean(input_train_,1);
% [mypca_train,eigV] = myPCA(input_train_,k,mF);
% mypca_test = calculate_embedding('pca',input_test_,eigV,mF);
% [m,n] = size(mypca_train);
% [guess_label,test_acc] = test_accuracy_(mypca_test,label_test,mypca_train,label_train);
% fprintf("PCA acc = %f\n",test_acc);

%LDA准确性
[D,mylda_train,W] = myLDA(input_train_,label_train,k);
mylda_test = calculate_embedding('lda',input_test_,W);
[guess_label,test_acc] = test_accuracy_(mylda_test,label_test,mylda_train,label_train);
fprintf("LDA acc = %f\n",test_acc);

%PCA+LDA
% [mypca_train,eigV] = myPCA(input_train_,30,mF);
% mypca_test = calculate_embedding('pca',input_test_,eigV,mF);
% 
% [D,mylda_train,W] = myLDA(mypca_train,label_train,k);
% mylda_test = calculate_embedding('lda',mypca_test,W);
% [guess_label,test_acc] = test_accuracy_(mylda_test,label_test,mylda_train,label_train);
% fprintf("PCA + LDA acc = %f\n",test_acc);


function [guess_label,acc] = test_accuracy_(test_img,test_label,train_img,train_label)
[m,n] = size(test_img); %fprintf("test_img m = %d and n = %d\n",m,n);
% [m2,n2] = size(train_img); fprintf("train_img m = %d and n = %d\n",m2,n2);
% [m3,n3] = size(test_label); fprintf("test_label m = %d and n = %d\n",m3,n3);
% [m4,n4] = size(train_label); fprintf("train_label m = %d and n = %d\n",m4,n4);
guess_label = zeros(m,1);
for i = 1:m
    tmp = calculate_label(test_img(i,:),train_img,train_label);
    guess_label(i) = tmp;
end
delta_test = guess_label - test_label;
acc_num = sum(delta_test(:)==0);
acc = double(acc_num) / double(m);
end

function [output] = calculate_label(img,dataset,label)
    [m,n] = size(dataset);
    dis_list = zeros(m,1);
    for i=1:m
        dis_list(i) = dis_L2(img,dataset(i,:));
    end
    [row,col] = find(dis_list==min(min(dis_list)));
    output = label(min(row),1);
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