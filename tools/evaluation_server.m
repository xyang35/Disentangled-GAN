
%root = '/home/xyang/Downloads/GAN/disentangled_resnet_9blocks_sigmoid_A100_TV0.00001/disentangled_resnet_9blocks_sigmoid_A100_TV0.00001/test_latest/images/';
%name = 'disentangled_resnet_9blocks_sigmoid_A100_TV1';

root = ['/home-4/xyang35@umd.edu/work/xyang/GAN/Haze/D-HAZY/results/',name,'/test_latest/images/'];

%suffix = '_dcp_radiance-refinedt'; folder = ['DCP/',name,'/']; 
%suffix = '_DehazeNet'; folder = ['DehazeNet/',name,'/'];
suffix = '_Haze-free'; folder = root;

img_names = dir([root, '/*_Hazy_real_B.png']);

peaksnr_all = zeros(length(img_names),1);
ssim_all = zeros(length(img_names),1);

display(folder);
display(suffix);

%parpool(4)
%parfor i = 1 : length(img_names)
for i = 1 : length(img_names)
    display( img_names(i).name );
    ref_img = imread([root, img_names(i).name]);
    name = strrep(img_names(i).name, '_real_B.png', [suffix,'.png']);
    img = imread([folder, name]);
    
    [peaksnr, snr] = psnr(img, ref_img);
    peaksnr_all(i) = peaksnr;
    
    ssim_all(i) = ssim(img, ref_img);
end

save([folder, 'evaluation.mat'], 'peaksnr_all', 'ssim_all', 'img_names');

display( mean(peaksnr_all) );
display( mean(ssim_all) );
