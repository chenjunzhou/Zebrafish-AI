function CV2(srcDir, output_xlsx_path)

srcDir = 'C:\Users\Dell\Desktop\20230731\Plate1-20230731_segment-datasets\Plate1-20230731_Ó«¹â\CCV\img_region';
output_xlsx_path = 'D:\ZZY matlab 20231209\·Ö¸îÍ¼\CVP.xls';
if exist('srcDir', 'var')
    srcDir=uigetdir('');
end
format long g;  
cd(srcDir);

allnames=struct2cell(dir('*.PNG')); 
if(isempty(allnames))
    allnames=struct2cell(dir('*.bmp'));
end
[k,len]=size(allnames);
image_name=cell(len,1);


N= 'Sample name';
Hn='Loop number';
Vs= 'Vessel area';
Ps= 'Perimeter';
Ck='Aspect ratio ';
Sp = 'Rectangularity';
F='Compactness';
Dens='Vessel density';
Y= {N,Hn,Vs,Ps,Ck,Sp,F,Dens};
for ii=1:len
    name=allnames{1,ii};
    I=imread(name); 
    image_name{ii,1}=name;
    X = CV2_one_image(I);

    range=sprintf('B%d:H%d',ii+1,ii+1);
    xlswrite(output_xlsx_path,X,range);
    xlswrite(output_xlsx_path,image_name,sprintf('A2:A%d',ii+1));

end
xlswrite(output_xlsx_path,Y,'A1:H1');
end

function feature = CV2_one_image(I)
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


B=rgb2gray(I);

T=0.3*(double(min(B(:)))+double(max(B(:))));
d=false;
while~d
     g=B>=T;
     Tn=0.3*(mean(B(g))+mean(B(~g)));
     d=abs(T-Tn)<0.3;
     T=Tn;
end
level=Tn/255;

BW=im2bw(B,level);
sehj0=strel('disk',1);
sehj00=strel('disk',1);
BW=imerode(BW,sehj0);
BW=imdilate(BW,sehj00);

BW=bwareaopen(BW,500,8);
BWH=imfill( BW , 'holes');

%feature
H= BWH -BW;
H2 = bwareaopen(H,20,8);
num=max(max(bwlabel(H2))); %Loop number
H2=H2-0;
BW2=BWH-H2;
BW2= BW2-0;
BWH=BWH-0;
S=regionprops(BW2, 'Area'); 
S=cat(1,S. Area); %Vessel area

P=regionprops(J, 'Perimeter');
P=cat(1, P. Perimeter); %perimeter
[r c]=find(J ==1);
[rectx,recty,area,perimeter] = minboundrect (c,r,'p');
dd = [rectx(1:end-1),recty(1:end-1)];
dd1 = dd([4 1 2 3],:);
ds = sqrt(sum((dd-dd1).^2,2));
kuan = min(ds(1:2)); 
chang= max(ds(1:2)); 
ck=chang/kuan;%Aspect ratio
Sq=kuan*chang; 
Spercent= S/Sq;%Rectangularity
F= (P ^2)/ (4 * pi *S ); %Compactness
St=regionprops(J, 'Area');   
St=cat(1,St. Area); 
Den=S/St; %Vessel density

feature ={num2str(num),num2str(S), num2str(P),num2str(ck),num2str(Spercent),num2str(F),num2str(Den)};

end

