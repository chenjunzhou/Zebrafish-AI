function CBV_function_all(srcDir, output_xlsx_path)
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
    I=CBV_Fluorescent(J);
    image_name{ii,1}=name;
    if(sum(sum(I~=0))~=0)
        [out_data(ii,1)]=CBV_Vessel_area(J);
        branch_point_15=CBV_branch_point(J);
        branch_point_100=CBV_branch_point_100(J);
        branch_number = branch_point_15-branch_point_100;
        out_data(ii,2) = branch_number; 
        fprintf('Branch number£º %d\n', out_data(ii,2));
    end
end
N= 'Sample name';
Vs = 'Vessel area' ;
bracnch_num = 'Branch number';
Y={N,Vs,bracnch_num};


xlswrite(output_xlsx_path,Y,'A1:C1');
xlswrite(output_xlsx_path,image_name,sprintf('A2:A%d',len+1));
xlswrite(output_xlsx_path,out_data,sprintf('B2:C%d',len+1));
end