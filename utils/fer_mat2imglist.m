load('../data/fer2013/fer2013.mat');
Xtr=permute(uint8(Xtr),[2,3,1]);
ytr=ytr';
Xtepub=permute(uint8(Xtepub),[2,3,1]);
ytepub=ytepub';
Xtepri=permute(uint8(Xtepri),[2,3,1]);
ytepri=ytepri';

file_train_path = '../data/fer2013/train.txt';
file_testpub_path = '../data/fer2013/PublicTest.txt';
file_testpri_path = '../data/fer2013/PrivateTest.txt';

counter = -1;
It=uint8(zeros(48,48,3));

% file = fopen(file_train_path,'w');
% for i = 1:28709
% counter = counter+1;
% It(:,:,1) = Xtr(:,:,i);
% It(:,:,2) = Xtr(:,:,i);
% It(:,:,3) = Xtr(:,:,i);
% imwrite(It,['img/fer',num2str(counter,'%0.5d'),'.jpg']);
% fprintf(file,['fer',num2str(counter,'%0.5d'),'.jpg ',num2str(ytr(i)),'\r\n']);
% if mod(i,1000)==0
% disp(i);
% end
% end

file = fopen(file_testpub_path,'w');
Xtr = Xtepub;
ytr = ytepub;
for i = 1:3589
    counter = counter+1;
    It(:,:,1) = Xtr(:,:,i);
    It(:,:,2) = Xtr(:,:,i);
    It(:,:,3) = Xtr(:,:,i);
    imwrite(It,['../data/fer2013/img/fer',num2str(counter,'%0.5d'),'.jpg']);
    fprintf(file,['fer',num2str(counter,'%0.5d'),'.jpg ',num2str(ytr(i)),'\r\n']);
    if mod(i,1000)==0
        disp(i);
    end
end

file = fopen(file_testpri_path,'w');
Xtr = Xtepri;
ytr = ytepri;
for i = 1:3589
    counter = counter+1;
    It(:,:,1) = Xtr(:,:,i);
    It(:,:,2) = Xtr(:,:,i);
    It(:,:,3) = Xtr(:,:,i);
    imwrite(It,['../data/fer2013/img/fer',num2str(counter,'%0.5d'),'.jpg']);
    fprintf(file,['fer',num2str(counter,'%0.5d'),'.jpg ',num2str(ytr(i)),'\r\n']);
    if mod(i,1000)==0
        disp(i);
    end
end
