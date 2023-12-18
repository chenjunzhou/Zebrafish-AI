function DA(srcDir, output_xlsx_path)
srcDir = 'C:\Users\Dell\Desktop\20230731\Plate1-20230731_segment-datasets\Plate1-20230731_Ó«¹â\DA\img_region';
output_xlsx_path = 'D:\ZZY matlab 20231209\·Ö¸îÍ¼\DA.xls';

if exist('srcDir', 'var')
    srcDir=uigetdir('');
end
format long g;
cd(srcDir);
allnames=struct2cell(dir('*.PNG'));
if(isempty(allnames))
    allnames=struct2cell(dir('*.png'));
end
[k,len]=size(allnames); 
image_name=cell(len,1);
N= 'Sample name';
W='Diameter';  
Y= {N,W};
for ii=1:len
    name=allnames{1,ii};
    I=imread(name); 
    image_name{ii,1}=name;
    X = DA_one_image(I);
    range=sprintf('B%d:B%d',ii+1,ii+1);
    xlswrite(output_xlsx_path,X,range);
    xlswrite(output_xlsx_path,image_name,sprintf('A2:A%d',ii+1));
end  


xlswrite(output_xlsx_path,Y,'A1:B1');
end


function feature = DA_one_image(I)

J=rgb2gray(I); 
[width,height,bmsize]=size(J); 

for i=1:width
    for j=1:height
        if J(i,j)>0
            J(i,j)=255;
        else
            J(i,j)=0;
        end
    end
end  

J=im2double(J); 
J=J-0;
S=regionprops(J, 'Area'); 
S=cat(1,S. Area); 

skel = bwmorph(J,'thin',Inf); 

bw5 = bwareaopen(skel, 256);
total_length=nnz(bw5); 
w = S/total_length;    

feature = {num2str(w)}; 
end




