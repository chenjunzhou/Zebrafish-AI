function [P,ck,Sol,Spercent,F] = SIV_other(input_img)
I=input_img;
J=I;
J=rgb2gray(I); 
J=im2double(J); 
sehj0=strel('disk',1);
sehj00=strel('disk',1);
J=imerode(J,sehj0);
J=imdilate(J,sehj00);
J=bwareaopen(J,200,8);
J=imfill(J,'hole');
J=J-0;

P=regionprops(J, 'Perimeter');
P=cat(1, P.Perimeter); 
[r, c]=find(J==1);
[rectx,recty,~,~] = minboundrect (c,r,'p');
dd = [rectx(1:end-1),recty(1:end-1)];
dd1 = dd([4 1 2 3],:);
ds = sqrt(sum((dd-dd1).^2,2));
kuan = min(ds(1:2));
chang= max(ds(1:2));
ck=chang/kuan;

Sj=regionprops(J, 'Area'); 
Sj=cat(1,Sj. Area); 
Sq=kuan*chang;
Spercent= Sj/Sq;

F= (P.^2)/ (4 * pi *Sj );

A = nnz(J); 
C=contour(J);
X=C(1,:); Y=C(2,:);
m=size(X,2);
mx=X(1); my=Y(1);
for i=2:m
    mx=mx+X(i); my=my+Y(i);
end
mx1=mx/m; my1=my/m;
max1=1; min1=99999;
for i=1:m
    d=((X(i)-mx1)^2+(Y(i)-my1)^2);
    if (d>max1)
        max1=d;
    end
    if (d<min1) 
        min1=d; 
    end
end

Sc=regionprops(J, 'ConvexArea');
Sc =cat(1, Sc. ConvexArea);
Sol=Sj/Sc;
end