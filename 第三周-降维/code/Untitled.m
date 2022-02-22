k1 = 2;
[mF,label_train,pcaF,eigV] = myeigen_face(num_person,1,k1);
[input_test,label_test] = read_face(num_person,0);
mypca_test = cal_latent('pca',input_test,eigV,mF);
%此时pcaF与label_train为训练;mypca_test为预测;label_test为实际,input_test为压缩
%我们之前已经测试过了pca方法的准确性，所以假设结果全部正确，再进行一次lda

k2 = 1;

[pcaF_train,eigV_train,mF_train] = myPCA(pcaF,k2);
[m,n] = size(mF_train);fprintf("mf size = %d * %d\n",m,n);
mypca_test2 = cal_latent('pca',mypca_test,eigV_train,mF_train);
[guess_label,test_acc] = test_accuracy_(mypca_test2,label_test,pcaF_train,label_train);
fprintf("PCA acc = %f\n",test_acc);

% [d_train,mylda_train,W] = lda(pcaF,label_train,k2);
% mylda_test = cal_latent('lda',mylda_train,W);
% [guess_label,test_acc2] = test_accuracy_(mylda_test,label_test,mylda_train,label_train);
% fprintf("LDA acc2 = %f\n",test_acc2);

function [output] = cal_latent(func_name,test_data,V,mA)
if strcmp(func_name,'pca')
    [m,n] = size(test_data);fprintf("test_data: m = %d and n = %d\n",m,n);
    [m,n] = size(V);fprintf("V_data: m = %d and n = %d\n",m,n);
    [m,n] = size(mA);fprintf("mA_data: m = %d and n = %d\n",m,n);
    output = (test_data-mA)*V;
elseif strcmp(func_name,'lda')
    output = test_data*V;
end
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
        dis_list(i) = dis_L1(img,dataset(i,:));
    end
    [row,col] = find(dis_list==min(min(dis_list)));
    fprintf("row_num = %d\n",row);
    fprintf("label = \n");disp(label);
    output = label(row);
    fprintf("output suc\n");
end
    
function [dis] = dis_L1(img1,img2)
delta_dis = img1 - img2;
dis = sum(abs(delta_dis(:)));
end

function [mF,label,eigF,eigV] = myeigen_face(num_person,flag,k)
global img_row img_col
  [orF,label] = read_face(num_person,flag);
  [pcaF,eigV,mF] = myPCA(orF,k);
  [m,n] = size(orF);
  eigF = pcaF;
end