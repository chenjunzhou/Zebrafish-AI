function sum = SIV_budding_num_2(I)
% J=I;
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
J= imdilate(J,sehj00); 
J=bwareaopen(J,200,8); 
J=imfill(J,'hole'); 

[m,n]=size(J);
J_left=J;
J_right=J;
[m,n]=size(J);


for i=1:m
    for j=1:n-1
        if(J_right(i,j)==1&&J_right(i,j+1)==1)
            J_right(i,j)=0;
        end
    end
end
for i=1:m
    for j=n:-1:2
        if(J_left(i,j)==1&&J_left(i,j-1)==1)
            J_left(i,j)=0;
        end
    end
end
for i=1:m
    for j=1:n
        if(J_right(i,j)==1 || J_left(i,j)==1)
            J(i,j)=1;
        else
            J(i,j)=0;
        end
    end
end

budding_num=0;
j=1;
while(j<=n-50)
    budding_num=budding_num+find_budding_num(J,j,m);
    j=j+30;
end

    function bud_num = find_budding_num(I,j,m)
        conut=0;bud_num=0;flag=0;
        ii=1;
        while(ii<m)
            if(j==1&&flag==0)
                ii=ii+30;
                flag=1;
            end
            left=0;right=0;help=0;
            for jj=j:j+50
                if(I(ii,jj)==1&&help==0)
                    left=jj;help=1;
                end
                if(I(ii,jj)==1)
                    right=jj;
                end
            end
            length=right-left;
            if(length<30&&length>0)
                conut=conut+1;
            end
            ii=ii+1;
        end
        if(conut>10)
            bud_num=1;
        end
    end
sum=budding_num;
end

