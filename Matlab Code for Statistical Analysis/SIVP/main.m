clc;
clear all;
warning off;

srcDir = 'C:\Users\Dell\Desktop\20230731\Plate1-20230731_segment-datasets\Plate1-20230731_ӫ��\CCV\img_region';
output_xlsx_path = 'D:\ZZY matlab 20231209\�ָ�ͼ\SIVP.xls';
tic;
SIV_function_all(srcDir, output_xlsx_path);
toc;
