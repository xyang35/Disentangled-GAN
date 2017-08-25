clear all;
clc

%root = '/home/xyang/Downloads/GAN/disentangled_resnet_9blocks_sigmoid_A100_TV0.00001/disentangled_resnet_9blocks_sigmoid_A100_TV0.00001/test_latest/images/';
name = 'disentangled_resnet_9blocks_sigmoid_A100_TV1';

root = ['/home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/',name,'/test_latest/images/'];
output_root = ['/home/xyang/Downloads/GAN/DehazeNet/', name, '/'];
mkdir (output_root);

img_names = dir([root, '*_Hazy_Hazy.png']);
for i = 1 : length(img_names)
    display(img_names(i).name);
    
    haze = imread([root, img_names(i).name]);
    haze = double(haze) ./ 255;
    [dehaze, F4] = run_cnn(haze);
    imwrite(dehaze, [output_root, strrep(img_names(i).name, '_Hazy.png', '_DehazeNet.png')]);
    imwrite(F4, [output_root, strrep(img_names(i).name, '_Hazy.png', '_transmition.png')]);
end