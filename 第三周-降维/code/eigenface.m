global img_row img_col
k = 20;
[pcaF,eigV] = eigen_face(41,1,k);
[m,n] = size(pcaF);
for i = 1:m
    eigen_data = pcaF(i,:);
    eigen_data = reshape(eigen_data,img_row,img_col);
    my_name = sprintf("output2\\%d.png",i);
    imwrite(uint8(eigen_data),char(my_name));
end
function [eigF,eigV] = eigen_face(num_person,flag,k)
global img_row img_col
  [orF,~] = read_face(num_person,flag);
  [pcaF,eigV,mF] = myPCA(orF,k);
  [m,n] = size(orF);
  eigF = zeros(m,img_row*img_col);
  for i1 =1:m
      eigF(i1,:) = pcaF(i1,:)*eigV'+mF;
  end
end