% Train a SVM on extracted features.
% paramters needed:
% alldata, 1: combine train and valid set for training; 0: Only train set will be used for training.
% savemodel, 1: savemodel to model/; 0:Do not save, use 0 during parameter tuning
% g,c, parameters of RBF kernel and loss function of SVM
%
% Note that: script will be EXTREMLY slow when savemodel=1, especially when dataset is augmentated. Use dimensional reduction method like PCA may help.
%
% Wu.


addpath('./libsvm-3.22/matlab');

if isempty(alldata)||isempty(savemodel)||isempty(g)||isempty(c)
    disp('Parameters required.');
end

if strcmp(method,'HOG')
    
    load('./data/feature/HOG.mat');
    Xtr_HOG = Xtr_HOG;
    ytr_HOG = ytr_HOG;
    Xte_HOG = Xte_HOG;
    yte_HOG = yte_HOG;
    if alldata
        Xtr_HOG = [Xtr_HOG;Xte_HOG];
        ytr_HOG = [ytr_HOG;yte_HOG];
    end
    if savemodel
        model = svmtrain(ytr_HOG,Xtr_HOG,['-b 1 -q -g ',num2str(g),' -c ',num2str(c)]);
        save model/SVM_HOG.mat model
    else
        model = svmtrain(ytr_HOG,Xtr_HOG,['-q -g ',num2str(g),' -c ',num2str(c)]);
        [y,acc,prob] = svmpredict(yte_HOG,Xte_HOG,model,'-q');
        disp(['acc: ', num2str(acc(1)),', with paramters:']);
        disp(['method: ',method]);
        disp([' -g ',num2str(g),' -c ',num2str(c)]);
    end
end

if strcmp(method,'dsift')
    %dsift
    
    load('./data/feature/dsift.mat');
    Xtr_HOG = Xtr_dsift;
    ytr_HOG = ytr_dsift;
    Xte_HOG = Xte_dsift;
    yte_HOG = yte_dsift;
    if alldata
        Xtr_HOG = [Xtr_HOG;Xte_HOG];
        ytr_HOG = [ytr_HOG;yte_HOG];
    end
    if savemodel
        model = svmtrain(ytr_HOG,Xtr_HOG,['-b 1 -q -g ',num2str(g),' -c ',num2str(c)]);
        save model/SVM_dsift.mat model
    else
        model = svmtrain(ytr_HOG,Xtr_HOG,['-q -g ',num2str(g),' -c ',num2str(c)]);
    end
    [y,acc,prob] = svmpredict(yte_HOG,Xte_HOG,model,'-q');
    disp(['acc: ', num2str(acc(1)),', with paramters:']);
    disp(['method: ',method]);
    disp([' -g ',num2str(g),' -c ',num2str(c)]);
    
end

if strcmp(method,'LBP')
    %LBP
    
    load('./data/feature/LBP.mat');
    Xtr_HOG = Xtr_LBP;
    ytr_HOG = ytr_LBP;
    Xte_HOG = Xte_LBP;
    yte_HOG = yte_LBP;
    if alldata
        Xtr_HOG = [Xtr_HOG;Xte_HOG];
        ytr_HOG = [ytr_HOG;yte_HOG];
    end
    if savemodel
        model = svmtrain(ytr_HOG,Xtr_HOG,['-b 1 -q -g ',num2str(g),' -c ',num2str(c)]);
        save model/SVM_LBP.mat model
    else
        model = svmtrain(ytr_HOG,Xtr_HOG,['-q -g ',num2str(g),' -c ',num2str(c)]);
    end
    [y,acc,prob] = svmpredict(yte_HOG,Xte_HOG,model,'-q');
    disp(['acc: ', num2str(acc(1)),', with paramters:']);
    disp(['method: ',method]);
    disp([' -g ',num2str(g),' -c ',num2str(c)]);
    
end

if strcmp(method,'LPQ')
    %LPQ
    
    load('./data/feature/LPQ.mat');
    Xtr_HOG = Xtr_LPQ;
    ytr_HOG = ytr_LPQ;
    Xte_HOG = Xte_LPQ;
    yte_HOG = yte_LPQ;
    if alldata
        Xtr_HOG = [Xtr_HOG;Xte_HOG];
        ytr_HOG = [ytr_HOG;yte_HOG];
    end
    if savemodel
        model = svmtrain(ytr_HOG,Xtr_HOG,['-b 1 -q -g ',num2str(g),' -c ',num2str(c)]);
        save model/SVM_LPQ.mat model
    else
        model = svmtrain(ytr_HOG,Xtr_HOG,['-q -g ',num2str(g),' -c ',num2str(c)]);
    end
    [y,acc,prob] = svmpredict(yte_HOG,Xte_HOG,model,'-q');
    disp(['acc: ',num2str(acc(1)),', with paramters:']);
    disp(['method: ',method]);
    disp([' -g ',num2str(g),' -c ',num2str(c)]);
    
    
end
