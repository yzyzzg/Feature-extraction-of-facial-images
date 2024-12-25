% 设置文件夹路径
input_folder = 'rawdata';  % 二进制文件夹路径
output_folder = 'output_images';  % 输出的 JPG 文件夹路径

% 如果输出文件夹不存在，则创建
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% 获取所有二进制文件的文件名
file_list = dir(fullfile(input_folder, '*'));  % 假设文件扩展名为 .bin

% 遍历文件夹中的所有文件
for i = 1:length(file_list)
    % 获取文件路径
    file_path = fullfile(input_folder, file_list(i+2).name);
    
    % 打开二进制文件并读取数据
    fid = fopen(file_path, 'rb');
    I = fread(fid);  % 读取为 uint8 格式
    fclose(fid);
    
    % 假设每个图像的尺寸是 128x128
    I = reshape(I, 128, 128)';  % 重新排列为 128x128 大小的矩阵
    I_gray = mat2gray(I);  % 将图像矩阵归一化到 [0, 1] 范围
        cmap = gray(256);      % 使用灰度色彩映射
        I_mapped = ind2rgb(uint8(I_gray * 255), cmap);  % 转为 RGB 格式
    
    % 保存为 JPG 文件
    [~, file_name, ~] = fileparts(file_list(i+2).name);  % 获取文件名（不带扩展名）
    output_path = fullfile(output_folder, [file_name, '.jpg']);
    imwrite(I_mapped, output_path);  % 保存为 JPG 格式
    
    fprintf('Saved %s as JPG\n', file_name);
end
