function [EulerNumber,totle_area_piex,Den,Vs] = SIV_hole_area(img_input)
I=img_input;
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
sehj0=strel('disk',1);
sehj00=strel('disk',1);
J=imerode(J,sehj0);
J=imdilate(J,sehj00);
J=bwareaopen(J,200,8);
J=imfill(J,'hole');

f=im2double(I);
T=0.4*(min(f(:))+max(f(:)));
done=false;
while ~done
	g=f>=T;
	Tn=0.4*(mean(f(g))+mean(f(~g)));
	done=abs(T-Tn)<0.1;
	T=Tn;
end
vv=im2bw(f,T);

vw=bwareaopen(vv,40,8); 
v=bwareaopen(vv,500,8); 
seh0=strel('disk',1);
seh00=strel('disk',2);
v=imerode(v,seh0);
v=imdilate(v,seh00);
vfh=imfill(v,'hole');
hz=vfh-v;
h=bwareaopen(hz,500,8);
v=vfh-h;
Hz=J-vw;
seh2=strel('disk',1);
Hz=imerode(Hz,seh2);

Hz=bwareaopen(Hz,100,8);
seh1=strel('disk',2);
Hz=imerode(Hz,seh1);
Hz=bwareaopen(Hz,50,8);
J=J-0;
v=v-0;
num=max(max(bwlabel(Hz))); 
Sj=regionprops(J, 'Area'); 
Sj=cat(1,Sj.Area); 
Sv=regionprops(v, 'Area'); 
Sv=cat(1,Sv.Area); 
Den=Sv/Sj;

EulerNumber=num;
totle_area_piex=Sj;
Vs=Sv;
Den=Den;
fprintf('Loop number£º         %d\n', EulerNumber);
fprintf('Region area£º     %d\n', totle_area_piex);
fprintf('Vessel area£º %d\n', Vs);
end
