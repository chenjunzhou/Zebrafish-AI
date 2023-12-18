function SIV_function_all(srcDir, output_xlsx_path)
if exist('srcDir', 'var')
    srcDir=uigetdir('');
end

fullpath = mfilename('fullpath');
[path,~]=fileparts(fullpath);

cd(srcDir);
allnames=struct2cell(dir('*.PNG')); 
if(isempty(allnames))
    allnames=struct2cell(dir('*.bmp'));
end

out_data=zeros(1000,18);
[~,len]=size(allnames); 
image_name=cell(len,1);
cd(path);
addpath('uigetdir');

for ii=1:len
    name=allnames{1,ii};
    J=imread([srcDir,'\',name]); 
    I=SIV_Fluorescent(J);
    image_name{ii,1}=name;
    
    fprintf('Picture:%s\n', ii,name);
   
    if(sum(sum(I~=0))~=0)
        [out_data(ii,1),out_data(ii,2),out_data(ii,3),out_data(ii,4)]=SIV_hole_area(J);
        out_data(ii,5)=SIV_budding_num_2(J);
        [out_data(ii,6),out_data(ii,7),out_data(ii,8),out_data(ii,9),...
            out_data(ii,10)]=SIV_other(J);
        
        out_data(ii,6)=roundn(out_data(ii,6),-4);
        out_data(ii,7)=roundn(out_data(ii,7),-2);
        out_data(ii,8)=roundn(out_data(ii,8),-2);
        out_data(ii,9)=roundn(out_data(ii,9),-2);
        out_data(ii,10)=roundn(out_data(ii,10),-2);
        out_data(ii,11)=roundn(out_data(ii,11),-2);
    end
end
N= 'Sample name';
Hn='Loop number';
Vs= 'Vessel area';
Ssiv='Region area';
budding_num='Leading bud number';
Den='Vessel density';
Ps= 'Perimeter';
Ck='Aspect ratio ';
So='Solidity';
Sp='Rectangularity';
Fs='Compactness';
Y={N,Hn,Ssiv,Den,Vs,budding_num,Ps,Ck,So,Sp,Fs};

xlswrite(output_xlsx_path,Y,'A1:K1');
xlswrite(output_xlsx_path,image_name,sprintf('A2:A%d',len+1));
xlswrite(output_xlsx_path,out_data,sprintf('B2:K%d',len+1));
end