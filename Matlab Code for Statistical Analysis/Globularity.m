function ratio=Globularity(ContourImage)
ContourImage_2=rgb2gray(ContourImage);

ContourImage_2=imfill(ContourImage_2,'hole');
ContourImage_2=bwperim(ContourImage_2);
for iii=5:2:15
    se=strel('disk',iii');
    ContourImage_3=imdilate(ContourImage_2,se);
    ContourImage_3=imfill(ContourImage_3,'hole');
    ContourImage_3=bwperim(ContourImage_3);
    ContourImage=uint8(ContourImage_3);
    [m,n]=size(ContourImage);
    for i=1:m
        for j=1:n
            if ContourImage(i,j)>0
                ContourImage(i,j)=255;
            end
        end
    end
    flag=0;
    for i=1:m
        if ContourImage(i,1)==255
            flag=flag+1;
        end
        if rem(flag,2)~=0
            ContourImage(i,1)=255;
        end
    end
    flag=0;
    for i=1:m
        if ContourImage(i,n)==255
            flag=flag+1;
        end
        if rem(flag,2)~=0
            ContourImage(i,n)=255;
        end
    end
    flag=0;
    for j=1:n
        if ContourImage(1,j)==255
            flag=flag+1;
        end
        if rem(flag,2)~=0
            ContourImage(1,j)=255;
        end
    end
    flag=0;
    for j=1:n
        if ContourImage(m,j)==255
            flag=flag+1;
        end
        if rem(flag,2)~=0
            ContourImage_2(m,j)=255;
        end
    end

    sz=size(ContourImage);
    [Y,X]=find(ContourImage==255,1, 'first');
    contour = bwtraceboundary(ContourImage, [Y(1), X(1)], 'W', 8);
    X=contour(:,2);
    Y=contour(:,1);

    BW=bwdist(logical(ContourImage));
    [Mx, My]=meshgrid(1:sz(2), 1:sz(1));
    [Vin Von]=inpoly([Mx(:),My(:)],[X,Y]);
    ind=sub2ind(sz, My(Vin),Mx(Vin));
    [R RInd]=max(BW(ind));
    R=R(1);
    RInd=RInd(1); 
    [cy cx]=ind2sub(sz, ind(RInd));
    if R~=0
        break;
    end


end

plot_x=zeros(1,1);        
plot_y=zeros(1,1);
plot_x(1) = cy;
plot_y(1) = cx;
[pos1(:,1),pos1(:,2)]  = find(ContourImage_3>0);           
d = sqrt((pos1(:,1)-plot_x).^2 + (pos1(:,2) - plot_y).^2); 
R2 = max(d);    
ratio = R/R2;   
end
