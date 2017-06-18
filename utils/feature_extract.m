img_path = '../data/img/';
feature_path = '../data/feature/';
file_train_path = '../data/train.txt';
file_test_path = '../data/test.txt';

%Num_train = 1882;Num_test = 482;
Num_train = 9105;Num_test = 466;

if isempty(method)||strcmp(method,'all')
    method2 = 'all';
else
    method2 = 'not all';
    disp(method)
end

warning off

if strcmp(method,'HOG')||strcmp(method2,'all')
    %Extract HOG features (On Grid)
    method = 'HOG';
    Xtr_HOG = zeros(Num_train,1764);
    ytr_HOG = zeros(Num_train,1);
    file_train = fopen(file_train_path);
    for counter = 1:Num_train
        tline = fgetl(file_train);
        split_result = strsplit(tline,' ');
        target = [img_path, split_result{1}];
        label = str2num(split_result{2});
        I = rgb2gray(imread(target));
        I_64by64 = imresize(I,[64, 64]);
        Xtr_HOG(counter,:) = extractHOGFeatures(I_64by64,'CellSize',[8,8],'BlockSize',[2,2],'BlockOverlap',[1,1],'NumBins',9);
        ytr_HOG(counter) = label;
    end
    Xte_HOG = zeros(Num_test,1764);
    yte_HOG = zeros(Num_test,1);
    file_test = fopen(file_test_path);
    for counter = 1:Num_test
        tline = fgetl(file_test);
        split_result = strsplit(tline,' ');
        target = [img_path, split_result{1}];
        label = str2num(split_result{2});
        I = rgb2gray(imread(target));
        I_64by64 = imresize(I,[64, 64]);
        Xte_HOG(counter,:) = extractHOGFeatures(I_64by64,'CellSize',[8,8],'BlockSize',[2,2],'BlockOverlap',[1,1],'NumBins',9);
        yte_HOG(counter) = label;
    end
    HOG_mean = mean(Xtr_HOG);
    HOG_std = std(Xtr_HOG)+1e-80;
    %Xtr_HOG = (Xtr_HOG-ones(Num_train,1)*HOG_mean)./(ones(Num_train,1)*HOG_std);
    %Xte_HOG = (Xte_HOG-ones(Num_test,1)*HOG_mean)./(ones(Num_test,1)*HOG_std);
    save([feature_path, method, '_transparam.mat'], [method, '_mean'], [method, '_std']);
    save([feature_path, method, '.mat'],['Xtr_', method],['ytr_', method],['Xte_', method],['yte_', method]);
    disp([feature_path, method, '.mat: ','Xtr_', method,', ytr_', method,', Xte_', method,', yte_', method]);
end



if strcmp(method,'dsift')||strcmp(method2,'all')
    %Extract DenseSIFT features (On Grid)
    %Num_train = 346;Num_test = 89;
    method = 'dsift';
    Xtr_dsift = zeros(Num_train,6272);
    ytr_dsift = zeros(Num_train,1);
    file_train = fopen(file_train_path);
    for counter = 1:Num_train
        tline = fgetl(file_train);
        split_result = strsplit(tline,' ');
        target = [img_path, split_result{1}];
        label = str2num(split_result{2});
        I = rgb2gray(imread(target));
        I_64by64 = imresize(I,[64, 64]);
        [dense_sift_vector,~,~] = DenseSIFT(I_64by64,16,8);
        Xtr_dsift(counter,:) = reshape(dense_sift_vector,[1,6272]);
        ytr_dsift(counter) = label;
    end
    Xte_dsift = zeros(Num_test,6272);
    yte_dsift = zeros(Num_test,1);
    file_test = fopen(file_test_path);
    for counter = 1:Num_test
        tline = fgetl(file_test);
        split_result = strsplit(tline,' ');
        target = [img_path, split_result{1}];
        label = str2num(split_result{2});
        I = rgb2gray(imread(target));
        I_64by64 = imresize(I,[64, 64]);
        [dense_sift_vector,~,~] = DenseSIFT(I_64by64,16,8);
        Xte_dsift(counter,:) = reshape(dense_sift_vector,[1,6272]);
        yte_dsift(counter) = label;
    end
    dsift_mean = mean(Xtr_dsift);
    dsift_std = std(Xtr_dsift)+1e-80;
    %Xtr_dsift = (Xtr_dsift-ones(Num_train,1)*dsift_mean)./(ones(Num_train,1)*dsift_std);
    %Xte_dsift = (Xte_dsift-ones(Num_test,1)*dsift_mean)./(ones(Num_test,1)*dsift_std);
    save([feature_path, method, '_transparam.mat'], [method, '_mean'], [method, '_std']);
    save([feature_path, method, '.mat'],['Xtr_', method],['ytr_', method],['Xte_', method],['yte_', method]);
    disp([feature_path, method, '.mat: ','Xtr_', method,', ytr_', method,', Xte_', method,', yte_', method]);
end


if strcmp(method,'LPQ')||strcmp(method2,'all')
    %Extract LPQ features (On Grid)
    method = 'LPQ';
    methodf = @(I) lpq(I,3,0,1,'nh');
    %Num_train = 346;Num_test = 89;
    Xtr_LPQ = zeros(Num_train,12544);
    ytr_LPQ = zeros(Num_train,1);
    file_train = fopen(file_train_path);
    for counter = 1:Num_train
        tline = fgetl(file_train);
        split_result = strsplit(tline,' ');
        target = [img_path, split_result{1}];
        label = str2num(split_result{2});
        I = rgb2gray(imread(target));
        I_64by64 = imresize(I,[64,64]);
        Xtr_LPQ(counter,:) = h77(I_64by64,methodf,256);
        ytr_LPQ(counter) = label;
    end
    Xte_LPQ = zeros(Num_test,12544);
    yte_LPQ = zeros(Num_test,1);
    file_test = fopen(file_test_path);
    for counter = 1:Num_test
        tline = fgetl(file_test);
        split_result = strsplit(tline,' ');
        target = [img_path, split_result{1}];
        label = str2num(split_result{2});
        I = rgb2gray(imread(target));
        I_64by64 = imresize(I,[64,64]);
        Xte_LPQ(counter,:) = h77(I_64by64,methodf,256);
        yte_LPQ(counter) = label;
    end
    LPQ_mean = mean(Xtr_LPQ);
    LPQ_std = std(Xtr_LPQ)+1e-80;
    %Xtr_LPQ = (Xtr_LPQ-ones(Num_train,1)*LPQ_mean)./(ones(Num_train,1)*LPQ_std);
    %Xte_LPQ = (Xte_LPQ-ones(Num_test,1)*LPQ_mean)./(ones(Num_test,1)*LPQ_std);
    save([feature_path, method, '_transparam.mat'], [method, '_mean'], [method, '_std']);
    save([feature_path, method, '.mat'],['Xtr_', method],['ytr_', method],['Xte_', method],['yte_', method]);
    disp([feature_path, method, '.mat: ','Xtr_', method,', ytr_', method,', Xte_', method,', yte_', method]);
end


if strcmp(method,'LBP')||strcmp(method2,'all')
    %Extract LBP features (On Grid)
    method = 'LBP';
    mapping=getmapping(8,'u2');
    methodf = @(I) lbp(I,1,8,mapping,'nh');
    %Num_train = 346;Num_test = 89;
    Xtr_LBP = zeros(Num_train,2891);
    ytr_LBP = zeros(Num_train,1);
    file_train = fopen(file_train_path);
    for counter = 1:Num_train
        tline = fgetl(file_train);
        split_result = strsplit(tline,' ');
        target = [img_path, split_result{1}];
        label = str2num(split_result{2});
        I = rgb2gray(imread(target));
        I_64by64 = imresize(I,[64, 64]);
        Xtr_LBP(counter,:) = h77(I_64by64,methodf, 59);
        ytr_LBP(counter) = label;
    end
    Xte_LBP = zeros(Num_test,2891);
    yte_LBP = zeros(Num_test,1);
    file_test = fopen(file_test_path);
    for counter = 1:Num_test
        tline = fgetl(file_test);
        split_result = strsplit(tline,' ');
        target = [img_path, split_result{1}];
        label = str2num(split_result{2});
        I = rgb2gray(imread(target));
        I_64by64 = imresize(I,[64, 64]);
        Xte_LBP(counter,:) = h77(I_64by64,methodf, 59);
        yte_LBP(counter) = label;
    end
    LBP_mean = mean(Xtr_LBP);
    LBP_std = std(Xtr_LBP)+1e-80;
    %Xtr_LBP = (Xtr_LBP-ones(Num_train,1)*LBP_mean)./(ones(Num_train,1)*LBP_std);
    %Xte_LBP = (Xte_LBP-ones(Num_test,1)*LBP_mean)./(ones(Num_test,1)*LBP_std);
    save([feature_path, method, '_transparam.mat'], [method, '_mean'], [method, '_std']);
    save([feature_path, method, '.mat'],['Xtr_', method],['ytr_', method],['Xte_', method],['yte_', method]);
    disp([feature_path, method, '.mat: ','Xtr_', method,', ytr_', method,', Xte_', method,', yte_', method]);
end

warning on
disp('Done');