addpath('../libsvm-3.22/matlab')

load('../data/feature/HOG_prob.mat');
load('../data/feature/dsift_prob.mat');
load('../data/feature/LBP_prob.mat');
load('../data/feature/LPQ_prob.mat');
load('../data/feature/CNN_prob.mat');

if isempty(weight)
    weight = ones(5,1);
end

prob = weight(1)*prob_te_CNN+weight(2)*prob_te_HOG+weight(3)*prob_te_dsift+weight(4)*prob_te_LBP;%+weight(5)*prob_te_LPQ;

pred= argmax(prob);

acc = mean(pred==yte_CNN+1);

acc_HOG = mean(pred(1:89)==yte_CNN(1:89)+1);

disp(['weight: ',num2str(weight')]);
disp(['acc: ', num2str(acc), ' ,and on HOG: ', num2str(acc_HOG)]);

function pred = argmax(prob)
m = size(prob,1);
pred = -1*ones(m,1);
for i = 1:m
    [~,pred(i)] = max(prob(i,:));
end
end