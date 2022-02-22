global img_row img_col
k = 20;
num_person = 41;

data_process_method = 'lbp';
%PCA准确性
[mF,label_train,pcaF,eigV] = myeigen_face(num_person,1,k,data_process_method);
[input_test,label_test] = read_face(num_person,0);
[input_test_] = data_preprocess(input_test,data_process_method);
mypca_test = cal_latent('pca',input_test_,eigV,mF);
[m,n] = size(pcaF);
[guess_label,test_acc] = test_accuracy_(mypca_test,label_test,pcaF,label_train);
fprintf("PCA acc = %f\n",test_acc);

%LDA准确性

% [input2_train,label2_train] = read_face(num_person,1);
% [input2_train] = data_preprocess(input2_train,data_process_method);
% [mapped,mapping] = FisherLDA(input2_train,label2_train,k);
% 
% [input2_test,label2_test] = read_face(num_person,0);
% [input2_test] = data_preprocess(input2_test,data_process_method);
% mylda_test = cal_latent('lda',input2_test,mapping.M);
% [guess_label,test_acc2] = test_accuracy_(mylda_test,label2_test,mapped,label2_train);
% fprintf("LDA acc2 = %f\n",test_acc2);

%第二个LDA
% [input2_train,label2_train] = read_face(num_person,1);
% [input2_train_] = data_preprocess(input2_train,data_process_method);
% [input2_train_] = data_preprocess(input2_train_,'mean');
% 
% [sw,sb,jw] = lda2(input2_train_,label2_train,k);
% sw_1 = inv(sw);
% 
% [D,mylda,W] = lda(input2_train_,label2_train,k);
% 
% [input2_test,label2_test] = read_face(num_person,0);
% [input2_test_] = data_preprocess(input2_test,data_process_method);
% [input2_test_] = data_preprocess(input2_test_,'mean');
% 
% 
% mylda_test = cal_latent('lda',input2_test_,W);
% [guess_label,test_acc2] = test_accuracy_(mylda_test,label2_test,mylda,label2_train);
% fprintf("LDA acc2 = %f\n",test_acc2);


function [output] = cal_latent(func_name,test_data,V,mA)
if strcmp(func_name,'pca')
    output = (test_data-mA)*V;
elseif strcmp(func_name,'lda')
    output = test_data*V;
end

end

function [mF,label,eigF,eigV] = myeigen_face(num_person,flag,k,data_process_method)
global img_row img_col
  [orF,label] = read_face(num_person,flag);
  [orF_] = data_preprocess(orF,data_process_method);
  [pcaF,eigV,mF] = myPCA(orF_,k);
  eigF = pcaF;
end

function [guess_label,acc] = test_accuracy_(test_img,test_label,train_img,train_label)
[m,n] = size(test_img); fprintf("test_img m = %d and n = %d\n",m,n);
[m2,n2] = size(train_img); fprintf("train_img m = %d and n = %d\n",m2,n2);
[m3,n3] = size(test_label); fprintf("test_label m = %d and n = %d\n",m3,n3);
[m4,n4] = size(train_label); fprintf("train_label m = %d and n = %d\n",m4,n4);
guess_label = zeros(m,1);
for i = 1:m
    tmp = cal_label(test_img(i,:),train_img,train_label);
    %[x,y] = size(tmp); fprintf("tmp m = %d and n = %d\n",x,y);
    guess_label(i) = tmp;
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