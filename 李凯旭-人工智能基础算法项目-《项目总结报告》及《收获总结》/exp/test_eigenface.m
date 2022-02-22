global imgrow imgcol
npersons = 41;
k = 20;
flag = 0;
[eigF,eigV,class] = eigen_face(npersons,flag,k);
[m,n] = size(eigF);
for i = 1:m
    eigenface_img = eigF(i,:);
    eigenface_img = reshape(eigenface_img,[imgrow,imgcol]);
    img_name = sprintf("./output_eigenface/%d_%d.png",class(i),i - floor((i-1)/5) * 5);
    imwrite(uint8(eigenface_img),char(img_name));
end
function [eigF,eigV,class] = eigen_face(npersons,flag,k)
global imgrow imgcol
  [orF,class] = read_face(npersons,flag);
  mF = mean(orF,1);
  [pcaF,eigV] = myPCA(orF,k,mF);
  [m,~] = size(orF);
  eigF = zeros(m,imgrow*imgcol);
  for i1 =1:m
      eigF(i1,:) = pcaF(i1,:)*eigV'+mF;
  end
end