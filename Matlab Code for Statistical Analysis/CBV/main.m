clc;
clear all;
warning off;

srcDir='D:\ZZY matlab 20231209\分割图\CBV';
output_xlsx_path='D:\ZZY matlab 20231209\分割图\CBV.xls';
tic;
CBV_function_all(srcDir, output_xlsx_path);
toc;