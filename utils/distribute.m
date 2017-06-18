% Prepocessing of original datasets, distribute images into a uniform style:
% dataset_name/data/emotion/*.jpg
% five datasets are processed: JAFFE,CK+,KDEF,TFEID
% before run this script, raw datasets have to be ready at dataset_name/dataset_name_raw/
% reset path at line 13/14 before use
%
% 2017.6.16
% Wu


emotion_name =  {'Neutral','Angry','Contempt','Disgust','Fear','Happy','Sad','Surprise'};
emotion_short = {'NE','AN','CO','DI','FE','HA','SA','SU'};
data_set_path = '../data/';
caffe_path = '../caffe/matlab';

% find face using MTCNN
pdollar_toolbox_path=['../toolbox-master'];
MTCNN_path = ['../MTCNNv1/'];
caffe_model_path=[MTCNN_path, 'model'];
addpath(genpath(caffe_path));
addpath(genpath(pdollar_toolbox_path));
addpath(genpath(MTCNN_path));
caffe.set_mode_gpu();
caffe.set_device(0);
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

% JAFFE
disp('processing JAFFE');
for i = 1:8
    disp(['processing...',emotion_name{i}]);
    file_list = dir([data_set_path,'JAFFE/','jaffe_raw/*.',emotion_short{i},'*.tiff']);
    for j = 1:length(file_list)
        I = imread([data_set_path,'JAFFE/','jaffe_raw/', file_list(j).name]);
        [m,n,d] = size(I);
        if d==1
            I2 = uint8(zeros(m,n,3));
            I2(:,:,1)=I;I2(:,:,2)=I;I2(:,:,3)=I;
            I = I2;
        end
        [boudingboxes, points]=detect_face(I,minsize,PNet,RNet,ONet,threshold,false,factor);
        if size(boudingboxes,1)==0
            disp(['Error: MTCNN FAILE at ',data_set_path,'JAFFE/','jaffe_raw/', file_list(j).name]);
            continue
        end
        boudingboxes = floor(boudingboxes(1,:));
        boudingboxes(boudingboxes<1)=1;
        boudingboxes(3)=min(boudingboxes(3),n);
        boudingboxes(4) = min(boudingboxes(4),m);
        I2 = I(boudingboxes(2):boudingboxes(4),boudingboxes(1):boudingboxes(3),:);
        imwrite(I2,[data_set_path,'JAFFE/data/',emotion_name{i}, '/' , file_list(j).name,'.jpg']);
    end
end

% KDEF
disp('processing KDEF');
emotion_short = {'NE','AN','CO','DI','AF','HA','SA','SU'};
dir_list = dir([data_set_path,'KDEF/','KDEF_raw/']);
for k = 1:length(dir_list)
    for i = 1:8
        disp(['processing...',emotion_name{i}]);
        file_list = dir([data_set_path,'KDEF/','KDEF_raw/',dir_list(k).name,'/*',emotion_short{i},'S.JPG']);
        for j = 1:length(file_list)
            I = imread([data_set_path,'KDEF/','KDEF_raw/',dir_list(k).name,'/', file_list(j).name]);
            [m,n,d] = size(I);
            if d==1
                I2 = uint8(zeros(m,n,3));
                I2(:,:,1)=I;I2(:,:,2)=I;I2(:,:,3)=I;
                I = I2;
            end
            [boudingboxes, points]=detect_face(I,minsize,PNet,RNet,ONet,threshold,false,factor);
            if size(boudingboxes,1)==0
                disp(['Error: MTCNN FAILE at ',data_set_path,'JAFFE/','jaffe_raw/', file_list(j).name]);
                continue
            end
            boudingboxes = floor(boudingboxes(1,:));
            boudingboxes(boudingboxes<1)=1;
            boudingboxes(3)=min(boudingboxes(3),n);
            boudingboxes(4) = min(boudingboxes(4),m);
            I2 = I(boudingboxes(2):boudingboxes(4),boudingboxes(1):boudingboxes(3),:);
            imwrite(I2,[data_set_path,'KDEF/data/',emotion_name{i}, '/' , file_list(j).name]);
        end
    end
end

% CK+
disp('processing CK+');
emotion_name =  {'Neutral','Angry','Contempt','Disgust','Fear','Happy','Sad','Surprise'};
a = dir([data_set_path,'CK+','/Emotion/']);
a = a(3:end);
for k = 1:length(a)
    if mod(k,20)==0
        disp([num2str(k),'/',num2str(length(a))]);
    end
    p = [data_set_path,'CK+','/Emotion/', a(k).name, '/'];
    suba = dir(p);
    suba = suba(3:end);
    for t = 1:length(suba)
        txtfile = [p, suba(t).name, '/'];
        txtfile = dir(txtfile);
        if length(txtfile)==2
            continue
        end
        txtfile2 = txtfile(3).name;
        txtfile = [p, suba(t).name, '/', txtfile2];
        label = emotion_name(csvread(txtfile)+1);
        source = [data_set_path,'CK+','/cohn-kanade-images/', a(k).name,'/',suba(t).name,'/',txtfile2(1:end-12),'.png'];
        I = imread(source);
        [m,n,d] = size(I);
        if d==1
            I2 = uint8(zeros(m,n,3));
            I2(:,:,1)=I;I2(:,:,2)=I;I2(:,:,3)=I;
            I = I2;
        end
        [boudingboxes, points]=detect_face(I,minsize,PNet,RNet,ONet,threshold,false,factor);
        if size(boudingboxes,1)==0
            disp(['Error: MTCNN FAILE at ',data_set_path,'JAFFE/','jaffe_raw/', file_list(j).name]);
            continue
        end
        boudingboxes = floor(boudingboxes(1,:));
        boudingboxes(boudingboxes<1)=1;
        boudingboxes(3)=min(boudingboxes(3),n);
        boudingboxes(4) = min(boudingboxes(4),m);
        I2 = I(boudingboxes(2):boudingboxes(4),boudingboxes(1):boudingboxes(3),:);
        target = [data_set_path,'CK+','/data/', cell2mat(label), '/', txtfile2(1:end-12), '.jpg'];
        imwrite(I2,target);
    end
end
for k = 1:length(a)
    scource = [data_set_path,'CK+','/cohn-kanade-images/',a(k).name,'/001/',a(k).name,'_001_00000001.png'];
    target = [data_set_path,'CK+','/data/Neutral/',a(k).name,'_001_00000001.jpg'];
    try
        I = imread(scource);
        [m,n,d] = size(I);
        if d==1
            I2 = uint8(zeros(m,n,3));
            I2(:,:,1)=I;I2(:,:,2)=I;I2(:,:,3)=I;
            I = I2;
        end
        [boudingboxes, points]=detect_face(I,minsize,PNet,RNet,ONet,threshold,false,factor);
        if size(boudingboxes,1)==0
            disp(['Error: MTCNN FAILE at ',data_set_path,'JAFFE/','jaffe_raw/', file_list(j).name]);
            continue
        end
        boudingboxes = floor(boudingboxes(1,:));
        boudingboxes(boudingboxes<1)=1;
        boudingboxes(3)=min(boudingboxes(3),n);
        boudingboxes(4) = min(boudingboxes(4),m);
        I2 = I(boudingboxes(2):boudingboxes(4),boudingboxes(1):boudingboxes(3),:);
        imwrite(I2,target);
    end
end

% Collected
disp('processing Collected dataset');
emotion_short = {'neutral','angry','contempt','disgust','fear','happy','sadness','surprise'};
for i = 1:8
    disp(['processing...',emotion_name{i}])
    file_list = dir([data_set_path,'collected/','data_raw/',emotion_short{i},'/*.jpg']);
    for j = 21:length(file_list)
        %disp([data_set_path,'collected/','data_raw/',emotion_short{i},'/', file_list(j).name])
        I = imread([data_set_path,'collected/','data_raw/',emotion_short{i},'/', file_list(j).name]);
        [m,n,d] = size(I);
        if d==1
            I2 = uint8(zeros(m,n,3));
            I2(:,:,1)=I;I2(:,:,2)=I;I2(:,:,3)=I;
            I = I2;
        end
        % caffe.reset_all();
        % caffe.set_mode_gpu();
        % caffe.set_device(0);
        % threshold=[0.6 0.7 0.7];
        % factor=0.709;
        % prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
        % model_dir = strcat(caffe_model_path,'/det1.caffemodel');
        % PNet=caffe.Net(prototxt_dir,model_dir,'test');
        % prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
        % model_dir = strcat(caffe_model_path,'/det2.caffemodel');
        % RNet=caffe.Net(prototxt_dir,model_dir,'test');
        % prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
        % model_dir = strcat(caffe_model_path,'/det3.caffemodel');
        % ONet=caffe.Net(prototxt_dir,model_dir,'test');
        [boudingboxes, points]=detect_face(I,minsize,PNet,RNet,ONet,threshold,false,factor);
        if size(boudingboxes,1)==0
            disp(['Error: MTCNN FAILE at ',data_set_path,'JAFFE/','jaffe_raw/', file_list(j).name]);
            continue
        end
        boudingboxes = floor(boudingboxes(1,:));
        boudingboxes(boudingboxes<1)=1;
        boudingboxes(3)=min(boudingboxes(3),n);
        boudingboxes(4) = min(boudingboxes(4),m);
        I2 = I(boudingboxes(2):boudingboxes(4),boudingboxes(1):boudingboxes(3),:);
        imwrite(I2,[data_set_path,'collected/','data/',emotion_name{i},'/', file_list(j).name]);
    end
end
disp('Done');