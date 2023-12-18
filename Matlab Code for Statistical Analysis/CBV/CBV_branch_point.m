function [s] = CBV_branch_point(img_input)
I=img_input;
I=rgb2gray(I);
[m,n]=size(I);

bwskel = bwmorph(I,'thin',Inf);

for kk=1:15
    for i=1:m
        for j=1:n
            if bwskel(i,j)==1
                if(i-1>0&&i<m&&j-1>0&&j<n)
                left=bwskel(i-1,j)+bwskel(i+1,j)+bwskel(i-1,j-1)+bwskel(i,j-1)+bwskel(i+1,j-1)+bwskel(i-1,j+1)+bwskel(i,j+1)+bwskel(i+1,j+1);
                if left<=1
                    bwskel(i,j)=0;
                end
                end
            end
        end
    end
end



BP = bwmorph(bwskel,'branchpoints');

count=0;
for i=3:m-3
    for j=3:n-3
        if BP(i,j)~=0
            for ii=i-2:i+2
                for jj=j-2:j+2
                    count=count+bwskel(ii,jj);
                end
            end
            if count>6

            else
                BP(i,j)=0;
            end
            count=0;
        end
    end
end
s=nnz(BP);
end