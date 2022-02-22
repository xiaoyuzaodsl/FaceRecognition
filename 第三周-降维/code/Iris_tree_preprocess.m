function [attrib]=Iris_tree_preprocess(  )
%数据预处理
 [attrib1, attrib2, attrib3, attrib4, class] = textread('C:\Users\likaixu\Desktop\iris.data', '%f%f%f%f%s', 'delimiter', ',');
 % delimiter , 是跳过符号“，”
 a = zeros(150, 1); 
 a(strcmp(class, 'Iris-setosa')) = 1; 
 a(strcmp(class, 'Iris-versicolor')) = 2; 
 a(strcmp(class, 'Iris-virginica')) = 3; 
%% 导入鸢yuan尾花数据
for i=1:150
    attrib(i,1)=attrib1(i);
    attrib(i,2)=attrib2(i);
    attrib(i,3)=attrib3(i);
    attrib(i,4)=attrib4(i);
    attrib(i,5)=a(i);
end
 
end