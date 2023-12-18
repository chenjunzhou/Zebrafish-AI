function CCV(srcDir, output_xlsx_path)

srcDir = 'C:\Users\Dell\Desktop\20230731\Plate1-20230731_segment-datasets\Plate1-20230731_Ó«¹â\CCV\img_region';
output_xlsx_path = 'D:\ZZY matlab 20231209\·Ö¸îÍ¼\CCV.xls';

if exist('srcDir', 'var')
    srcDir=uigetdir('');
end
cd(srcDir);
allnames=struct2cell(dir('*.PNG')); 
if(isempty(allnames))
    allnames=struct2cell(dir('*.png'));
end
[k,len]=size(allnames);             

N= 'Sample';
Vs= 'Vessel area';
Ps= 'Perimeter';
Ck = 'Aspect ratio';
Ir = 'Irregularity';
IR = 'Roundness';
Sp = 'Rectangularity';

Y= {N,Vs,Ps,Ck,Ir,IR,Sp};
image_name=cell(len,1);

for ii=1:len  
    name=allnames{1,ii};
    I=imread(name);  
    image_name{ii,1}=name;
    X = CCV_one_image(I);
    range=sprintf('B%d:G%d',ii+1,ii+1);
    xlswrite(output_xlsx_path,X,range);
    xlswrite(output_xlsx_path,image_name,sprintf('A2:A%d',ii+1));
end  

xlswrite(output_xlsx_path,Y,'A1:G1');
end

function feature = CCV_one_image(I)
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
seh0=strel('disk',3);
seh00=strel('disk',3);
J=imerode(J,seh0);   
J=imdilate(J,seh00); 

J=bwareaopen(J,1000,8); 
J=imfill(J, 'hole');    
J=im2double(J); 
u=J;
u=u-0;
S=regionprops(u, 'Area'); 
S=cat(1,S. Area); %Vessel area
P=regionprops(u, 'Perimeter');
P=cat(1, P.Perimeter); %perimeter
[r c]=find(u==1);
[rectx,recty,area,perimeter] = minboundrect (c,r,'p');
dd = [rectx(1:end-1),recty(1:end-1)];
dd1 = dd([4 1 2 3],:);
ds = sqrt(sum((dd-dd1).^2,2));
kuan = min(ds(1:2));  
chang= max(ds(1:2));  
ck=chang/kuan;  %Aspect ratio      
Sq=kuan*chang;
Spercent= S/Sq;          

IR = Globularity(I);
[L,num]=bwlabel(J,8);      
plot_x=zeros(1,1);        
plot_y=zeros(1,1);
sum_x=0;sum_y=0;area=0;
[height,width]=size(J);
for i=1:height
    for j=1:width
        if L(i,j)==1
            sum_x=sum_x+i;
            sum_y=sum_y+j;
            area=area+1;
        end
    end
end

plot_x(1)=fix(sum_x/area);
plot_y(1)=fix(sum_y/area);
 contour = bwperim(J); 
 [pos1(:,1),pos1(:,2)]  = find(contour>0);   
d2 = (pos1(:,1)-plot_x).^2 + (pos1(:,2) - plot_y).^2; 
Ir=pi*max(d2)/S ;
feature = {num2str(S),num2str(P),num2str(ck),num2str(Ir),num2str(IR),num2str(Spercent)};
end


