switch_HOG = 0;
switch_LBP = 0;
switch_LPQ = 0;
switch_dsift = 0;
switch_CNN = 1;
fusion_weight = [1;1;1;1;1];
weights = 'model/mul3_iter_10000.caffemodel';
addpath('./libsvm-3.22/matlab');
addpath('caffe/matlab');
face_mean = caffe.io.read_mean('./data/face_mean_all.binaryproto');
test_path = '/home/litp/wzy/test2/data/dataset/test/collected';
output_file = 'result.txt';
emotion_name =  {'ne','an','co','di','fe','ha','sa','su'};
caffe.set_mode_gpu();
caffe.set_device(0);


addpath('./utils');
model = './model/VGG_FACE_finetune_deploy3.prototxt';
net = caffe.Net(model, weights, 'test');
pdollar_toolbox_path='./toolbox-master';
MTCNN_path = './MTCNNv1/';
caffe_model_path=[MTCNN_path, 'model'];
addpath(genpath(pdollar_toolbox_path));
addpath(genpath(MTCNN_path));
threshold=[0.6 0.7 0.7];
factor=0.709;
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
model_dir = strcat(caffe_model_path,'/det1.caffemodel');
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
model_dir = strcat(caffe_model_path,'/det2.caffemodel');
RNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
model_dir = strcat(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');
minsize=20;
if switch_HOG
    load('model/SVM_HOG.mat');model_HOG=model;
end
if switch_dsift
    load('model/SVM_dsift.mat');model_dsift=model;
end
if switch_LBP
    load('model/SVM_LBP.mat');model_LBP=model;
end
if switch_LPQ
    load('model/SVM_LPQ.mat');model_LPQ=model;
end

img_list = dir([test_path,'/*.jpg']);
file = fopen(output_file,'w');
for c = 1:length(img_list)
    disp(['processing...',num2str(c),'/',num2str(length(img_list))]);
    I = imread([test_path,'/',img_list(c).name]);
    [m,n,d] = size(I);
    if m*n>1500*1000
        I = imresize(I,floor([m/pi,n/pi]));
    end
    [m,n,d] = size(I);
    if d==1
        I2 = uint8(zeros(m,n,3));
        I2(:,:,1) = I;I2(:,:,2) = I;I2(:,:,3) = I;
        I = I2;
    end
    %MTCNN
    [boudingboxes, points]=detect_face(I,minsize,PNet,RNet,ONet,threshold,false,factor);
    if size(boudingboxes,1)==0
        disp(['MTCNN error when process ',img_list(c).name]);
    else
        boudingboxes = floor(boudingboxes(1,:));
        boudingboxes(boudingboxes<1)=1;
        boudingboxes(3)=min(boudingboxes(3),n);
        boudingboxes(4) = min(boudingboxes(4),m);
        I = I(boudingboxes(2):boudingboxes(4),boudingboxes(1):boudingboxes(3),:);
    end
    %HOG
    if switch_HOG
        I_64by64 = imresize(rgb2gray(I),[64, 64]);
        f_HOG = extractHOGFeatures(I_64by64,'CellSize',[8,8],'BlockSize',[2,2],'BlockOverlap',[1,1],'NumBins',9);
        [~,~,prob_HOG]=svmpredict(0.0,double(f_HOG),model_HOG,'-q -b 1');
    else
        prob_HOG = zeros(1,8);
    end
    %dsift
    if switch_dsift
        [f_dsift,~,~] = DenseSIFT(I_64by64,16,8);
        f_dsift = reshape(f_dsift,[1,6272]);
        [~,~,prob_dsift]=svmpredict(0.0,double(f_dsift),model_dsift,'-q -b 1');
    else
        prob_dsift = zeros(1,8);
    end
    %LPQ
    if switch_LPQ
        methodf = @(I) lpq(I,3,0,1,'nh');
        f_LPQ = h77(I_64by64,methodf,256);
        [~,~,prob_LPQ]=svmpredict(0.0,double(f_LPQ),model_LPQ,'-q -b 1');
    else
        prob_LPQ = zeros(1,8);
    end
    %LBP
    if switch_LBP
        mapping=getmapping(8,'u2');
        methodf = @(I) lbp(I,1,8,mapping,'nh');
        f_LBP = h77(I_64by64,methodf, 59);
        [~,~,prob_LBP]=svmpredict(0.0,double(f_LBP),model_LBP,'-q -b 1');
    else
        prob_LBP = zeros(1,8);
    end
    %CNN
    if switch_CNN
        imwrite(I,'temp.jpg');
        I3 = caffe.io.load_image('temp.jpg');
        I_224 = imresize(I3,[224,224]);
        res = net.forward({I_224-face_mean});
        prob_CNN = res{1}';
    else
        prob_CNN = zeros(1,8);
    end
    %fusion
    [~,pred]=max(fusion_weight(2)*prob_HOG+fusion_weight(3)*prob_dsift+fusion_weight(5)*prob_LPQ+fusion_weight(4)*prob_LBP+fusion_weight(1)*prob_CNN);
    pred = emotion_name{pred};
    fprintf(file,[img_list(c).name,' ',pred,'\r\n']);
    disp([img_list(c).name,' ',pred]);
end