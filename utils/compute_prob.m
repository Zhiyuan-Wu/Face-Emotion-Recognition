addpath('../libsvm-3.22/matlab');
addpath('../caffe/matlab');


switch_HOG = 1;
switch_LBP = 1;
switch_LPQ = 1;
switch_dsift = 1;
switch_CNN = 1;
weights = '../model/mul3_iter_3800.caffemodel';

if switch_HOG
    load('../data/feature/HOG.mat');
    load('../model/SVM_HOG.mat')
    [y,acc_tr,prob_tr_HOG] = svmpredict(ytr_HOG,Xtr_HOG,model,'-q -b 1');
    [y,acc_te,prob_te_HOG] = svmpredict(yte_HOG,Xte_HOG,model,'-q -b 1');
    save ../data/feature/HOG_prob.mat prob_tr_HOG ytr_HOG prob_te_HOG yte_HOG
    disp(['HOG alone, train acc: ',num2str(acc_tr(1)),' ,test acc: ',num2str(acc_te(1))])
end

if switch_dsift
    load('../data/feature/dsift.mat');
    load('../model/SVM_dsift.mat')
    [y,acc_tr,prob_tr_dsift] = svmpredict(ytr_dsift,Xtr_dsift,model,'-q -b 1');
    [y,acc_te,prob_te_dsift] = svmpredict(yte_dsift,Xte_dsift,model,'-q -b 1');
    save ../data/feature/dsift_prob.mat prob_tr_dsift ytr_dsift prob_te_dsift yte_dsift
    disp(['disft alone, train acc: ',num2str(acc_tr(1)),' ,test acc: ',num2str(acc_te(1))])
end

if switch_LBP
    load('../data/feature/LBP.mat');
    load('../model/SVM_LBP.mat')
    [y,acc_tr,prob_tr_LBP] = svmpredict(ytr_LBP,Xtr_LBP,model,'-q -b 1');
    [y,acc_te,prob_te_LBP] = svmpredict(yte_LBP,Xte_LBP,model,'-q -b 1');
    save ../data/feature/LBP_prob.mat prob_tr_LBP ytr_LBP prob_te_LBP yte_LBP
    disp(['LBP alone, train acc: ',num2str(acc_tr(1)),' ,test acc: ',num2str(acc_te(1))])
end

if switch_LPQ
    load('../data/feature/LPQ.mat');
    load('../model/SVM_LPQ.mat')
    [y,acc_tr,prob_tr_LPQ] = svmpredict(ytr_LPQ,Xtr_LPQ,model,'-q -b 1');
    [y,acc_te,prob_te_LPQ] = svmpredict(yte_LPQ,Xte_LPQ,model,'-q -b 1');
    save ../data/feature/LPQ_prob.mat prob_tr_LPQ ytr_LPQ prob_te_LPQ yte_LPQ
    disp(['LPQ alone, train acc: ',num2str(acc_tr(1)),' ,test acc: ',num2str(acc_te(1))])
end

if switch_CNN
    
    img_path = '../data/img/';
    file_train_path = '../data/train.txt';
    file_test_path = '../data/test.txt';
    face_mean = caffe.io.read_mean('../data/face_mean_aug.binaryproto');
    model = '../model/VGG_FACE_finetune_deploy3.prototxt';
    
    caffe.set_mode_gpu();
    net = caffe.Net(model, weights, 'test');
    Num_train = 9105;
    file_train = fopen(file_train_path);
    prob_tr_CNN = zeros(Num_train,8);
    ytr_CNN = zeros(Num_train,1);
    counter = 0;
    for counter = 1:Num_train
        %disp(['processing...',num2str(counter),'/',num2str(Num_train)]);
        tline = fgetl(file_train);
        split_result = strsplit(tline,' ');
        target = [img_path, split_result{1}];
        label = str2num(split_result{2});
        I = caffe.io.load_image(target);
        I_224 = imresize(I,[224,224]);
        res = net.forward({I_224-face_mean});
        prob_tr_CNN(counter,:) = res{1}';
        ytr_CNN(counter) = label;
    end
    Num_test = 466;
    file_test = fopen(file_test_path);
    prob_te_CNN = zeros(Num_test,8);
    yte_CNN = zeros(Num_test,1);
    counter = 0;
    for counter = 1:Num_test
        %disp(['processing...',num2str(counter),'/',num2str(Num_test)]);
        tline = fgetl(file_test);
        split_result = strsplit(tline,' ');
        target = [img_path, split_result{1}];
        label = str2num(split_result{2});
        I = caffe.io.load_image(target);
        I_224 = imresize(I,[224,224]);
        res = net.forward({I_224-face_mean});
        res = net.forward({I_224});
        prob_te_CNN(counter,:) = res{1}';
        yte_CNN(counter) = label;
    end
    save ../data/feature/CNN_prob.mat prob_tr_CNN ytr_CNN prob_te_CNN yte_CNN
    disp('CNN finish')
end