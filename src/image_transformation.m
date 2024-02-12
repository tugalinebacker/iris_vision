
%% IMAGE TRANSFORMATION
clear
clc

%% Parameters setup
BRIGHTNESS = 70;
CONTRAST = 0.5;
GAMMA = 1.7;
SIG_ALPHA = 5;
SIG_BETA = 2.6;

dataset = imageDatastore('~\LABELED_mkII\GHOST_NETS_mkII-PascalVOC-export\JPEGImages')
% n = numel(dir('~~\LABELED\vott-csv-export\*.jpg'));
n = length(dataset.Files);

%for i = 1:n
inc = 475
for i = 889:904
    %NEGATIVE BRIGHTNESS
    image = readimage(dataset,i);
    Bn = image-BRIGHTNESS;
   
    %SAVE BRIGHTNESS
%     filename_bn = sprintf('%s%d','neg_bright_',i)
%     imwrite(Bn,['~\TEST_DATASET_BR\' filename_bn '.jpg'])
    filename_bn = sprintf('%s%d','neg_bright_',inc)
    %imwrite(Bn,['~\TEST_DATASET_BR\' filename_bn '.jpg'])
    imwrite(Bn,['~\TEST_DATASET_INCREASE1\' filename_bn '.jpg'])
   
    %NEGATIVE CONTRAST
    image16=int16(Bn)-127;
    Clow = uint8(image16*CONTRAST+127);
   
    %POSITIVE GAMMA
    Dimage = double(Clow);
    GaH = (Dimage/255).^GAMMA;
   
    %NEGATIVE SIGMOID
    SigL = 1./(1+exp(-SIG_ALPHA*(double(Clow)/255)+SIG_BETA));
   
    %SAVE SIGMOID
%     filename_sig = sprintf('%s%d', 'sigmoid_',i);
%     imwrite(SigL,['~\TEST_DATASET_SIG\' filename_sig '.jpg'])
    filename_sig = sprintf('%s%d', 'sigmoid_',inc);
    %imwrite(SigL,['~\TEST_DATASET_SIG\' filename_sig '.jpg'])
    imwrite(SigL,['~\TEST_DATASET_INCREASE1\' filename_sig '.jpg'])
   
    inc=inc+1;
   
end