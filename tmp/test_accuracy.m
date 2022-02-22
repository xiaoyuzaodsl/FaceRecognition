global img_row img_col
k = 20;
num_person = 41;

%PCA准确性
% [mF,label_train,pcaF,eigV] = myeigen_face(num_person,1,k);
% [input_test,label_test] = read_face(num_person,0);
% mypca_test = cal_latent('pca',input_test,eigV,mF);
% [m,n] = size(pcaF);
% [guess_label,test_acc] = test_accuracy_(mypca_test,label_test,pcaF,label_train);
% fprintf("PCA acc = %f\n",test_acc);

%LDA准确性

[input2_train,label2_train] = read_face(num_person,1);
[d_train,mylda_train,W] = lda(input2_train,label2_train,k);

[input2_test,label2_test] = read_face(num_person,0);
mylda_test = cal_latent('lda',input2_test,W);
[guess_label,test_acc2] = test_accuracy_(mylda_test,label2_test,mylda_train,label2_train);
fprintf("LDA acc2 = %f\n",test_acc2);


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
        dis_list(i) = dis_L1(img,dataset(i,:));
    end
    [row,col] = find(dis_list==min(min(dis_list)));
    output = label(row);
end
    
function [dis] = dis_L1(img1,img2)
delta_dis = img1 - img2;
dis = sum(abs(delta_dis(:)));
end