%% Load train imagedataset into directory

dir_img = fullfile('datafolder1');

%% Load corresponding labelled train imagedataset into pixel directory

dir_pxl = fullfile('datafolder2');


%% Assign class labels and Label IDs

classNames = ["background","nuclei","ER","cyto"];
labelIDs   = [0 2 3 4];

%% Load validation dataset into image directory

dir_valid = fullfile('datafolder3');

%% Load corresponding labelled validation dataset into pixel directory

dir_pxlvalid = fullfile('datafolder4');


%% semantic segmentation network creation 

numFilters = 64;
filterSize = 3;
numClasses = 4;
semseglayers = [
    imageInputLayer([512 512 1])
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
    convolution2dLayer(1,numClasses);
    softmaxLayer()
    pixelClassificationLayer()
    ];
semsegtraining = combine(dir_img,dir_pxl);
validationData = combine(dir_valid,dir_pxlvalid)

%% Set training options

opt_semseg = trainingOptions('sgdm', ...'InitialLearnRate',0.01, ...'MaxEpochs',50, ...
    'MiniBatchSize',32,'Plots','training-progress', ValidationData = validationData, ValidationFrequency=5, Shuffle='every-epoch');


%% Train the network
net_semseg = trainNetwork (semsegtraining,semseglayers,opt_semseg);
%% Load test data and labeled test data
dir_tst = fullfile('datafolder5');

dir_tstpxl = fullfile('datafolder6');

pxl_1 = pixelLabelDatastore(dir_tstpxl,classNames,labelIDs);
pxl_2 = semanticseg(dir_tst, net_semseg,'MiniBatchSize',4,'WriteLocation',tempdir);
semsegeval = evaluateSemanticSegmentation(pxl_2,pxl_1);
