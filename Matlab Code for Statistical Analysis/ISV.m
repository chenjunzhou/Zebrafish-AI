 function ISV(srcDir, output_xlsx_path)

srcDir = 'C:\Users\Dell\Desktop\20230731\Plate1-20230731_segment-datasets\Plate1-20230731_Ó«¹â\CCV\img_region';
output_xlsx_path = 'D:\ZZY matlab 20231209\·Ö¸îÍ¼\ISV.xls';

if exist('srcDir', 'var')
    srcDir=uigetdir('');
end
format long g;
cd(srcDir);

% allnames=struct2cell(dir('*.BMP')); 
allnames=struct2cell(dir('*.PNG')); 
if(isempty(allnames))
    allnames=struct2cell(dir('*.png'));
end
[k,len]=size(allnames); 
image_name=cell(len,1);

N= 'Sample name';
Ck='Aspect ratio ';
Sp='Rectangularity ';
Vl = 'Vessel length';
DLAV = 'Total interval length';
Y= {N,Ck,Sp,Vl,DLAV};

for ii=1:len
    name=allnames{1,ii};
    I=imread(name);
    image_name{ii,1}=name;
    X = ISV_one_image(I);

    range=sprintf('B%d:E%d',ii+1,ii+1);
    xlswrite(output_xlsx_path,X,range);
    xlswrite(output_xlsx_path,image_name,sprintf('A2:A%d',ii+1));
end  

xlswrite(output_xlsx_path,Y,'A1:E1');
end

function feature = ISV_one_image(I)

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
[r, c]=find(J ==1);
[rectx,recty,area,perimeter] = minboundrect (c,r,'p');
dd = [rectx(1:end-1),recty(1:end-1)];
dd1 = dd([4 1 2 3],:);
ds = sqrt(sum((dd-dd1).^2,2));
kuan = min(ds(1:2)); 
chang= max(ds(1:2)); 
ck=chang/kuan; 
Sq=kuan*chang; 

sehj0=strel('disk',1);
sehj00=strel('disk',1);
J=imerode(J,sehj0); 
J=imdilate(J,sehj00); 
J=bwareaopen(J,200,8); 
J=imfill(J,'hole');
St=regionprops(J, 'Area'); 
St=cat(1,St. Area); 
Spercent= St/ Sq; 

B=rgb2gray(I); 

T=0.5*(double(min(B(:)))+double(max(B(:))));
d=false;
while~d
     g=B>=T;
     Tn=0.43*(mean(B(g))+mean(B(~g)));
     d=abs(T-Tn)<0.1;
     T=Tn;
end
level=Tn/255;

BW=imbinarize(B,level);
sehj0=strel('disk',1);
sehj00=strel('disk',1);
BW=imerode(BW,sehj0); 
BW=imdilate(BW,sehj00); 
BW=bwareaopen(BW,100,8); 
BW=BW-0; 
J=J-0;

BW1 = BW; 
ROI = J; 

[pos1(:,1),pos1(:,2)]  = find(ROI>0); 
d1 = pos1(:,1).^2 + pos1(:,2).^2;     
d2 = pos1(:,1).^2 + (pos1(:,2) - size(ROI,2)).^2; 
[~,ind1] = min(d1);        
[~,ind2] = min(d2);       
x1 = pos1(ind1,2);       
y1 = pos1(ind1,1);
x2 = pos1(ind2,2);        
y2 = pos1(ind2,1);
index = sum(cumsum(ROI~=0,1) == 0,1)+ 1;
index(index>size(ROI,1)) = nan;          
y = index(x1:x2);                       
x = x1:x2;                              

x11 = floor(linspace(x1,x2,75));
y11 = index(x11);
d=0;
for i =1:(length(x11)-1)
    d = d + sqrt((y11(i)-y11(i+1))^2+(x11(i)-x11(i+1))^2);
end
vessel_gap = ~BW1&ROI;                    
vessel_gap = imopen(vessel_gap,strel('disk',8)); %
vessel_gap = bwareaopen(vessel_gap,100);         
vessel_gap_morphology = imdilate(vessel_gap,strel('rectangle',[2,150]));  
vessel_gap_morphology = imfill(vessel_gap_morphology,'hole');
vessel_gap_morphology = imerode(vessel_gap_morphology,strel('rectangle',[1,150])); 
vessel_gap_morphology = imerode(vessel_gap_morphology,strel('disk',2));
vessel = ~vessel_gap&vessel_gap_morphology;   
mean_vessel_len = St/d;

feature = {num2str(ck),num2str(Spercent),num2str(mean_vessel_len),num2str(d)};
end


