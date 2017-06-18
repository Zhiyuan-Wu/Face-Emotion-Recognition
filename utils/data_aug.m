% read pre-processed image,
% do basic augmentation,
% split data into train/test randomly at 8:2
% and Generate file list for caffe
% result is saved to data/img/
emotion_name =  {'Neutral','Angry','Contempt','Disgust','Fear','Happy','Sad','Surprise'};
data_path_list = {'../data/CK+/data/','../data/JAFFE/data/','../data/KDEF/data/','../data/TFEID/data/','../data/collected/data/'};
for data_set_num = 1:5
    disp(['Processing ',data_path_list{data_set_num}]);
    data_path = data_path_list{data_set_num};
    for label = 1:8
        if strcmp(data_path,'../data/KDEF/data/')
            file_list = dir([data_path, emotion_name{label}, '/*.JPG']);
        else
            file_list = dir([data_path, emotion_name{label}, '/*.jpg']);
        end
        total_num = length(file_list);
        if total_num==0
            continue
        end
        train_num = floor(0.8*total_num);
        test_num = total_num-train_num;
        index = randperm(total_num);
        train_index = index(1:train_num);
        test_index = index(train_num+1:end);
        file = fopen('../data/train.txt','a');
        for i = 1:train_num
            I = imread([data_path, emotion_name{label}, '/', file_list(train_index(i)).name]);
            imwrite(I,['../data/img/','ori_',file_list(train_index(i)).name]);
            fprintf(file,['ori_', file_list(train_index(i)).name, ' ', num2str(label-1), '\r\n']);
            imwrite(I+uint8(20*randn(size(I))),['../data/img/','nos_',file_list(train_index(i)).name]);
            fprintf(file,['nos_', file_list(train_index(i)).name, ' ', num2str(label-1), '\r\n']);
            imwrite(imrotate(I,15,'bilinear','crop'),['../data/img/','rtl_',file_list(train_index(i)).name]);
            fprintf(file,['rtl_', file_list(train_index(i)).name, ' ', num2str(label-1), '\r\n']);
            imwrite(imrotate(I,-15,'bilinear','crop'),['../data/img/','rtr_',file_list(train_index(i)).name]);
            fprintf(file,['rtr_', file_list(train_index(i)).name, ' ', num2str(label-1), '\r\n']);
            imwrite(flipdim(I,2),['../data/img/','flp_',file_list(train_index(i)).name]);
            fprintf(file,['flp_', file_list(train_index(i)).name, ' ', num2str(label-1), '\r\n']);
            %copyfile([data_path, emotion_name{label},'/',file_list(train_index(i)).name],['../data/img/',file_list(train_index(i)).name]);
        end
        fclose(file);
        file = fopen('../data/test.txt','a');
        for i = 1:test_num
            copyfile([data_path, emotion_name{label},'/',file_list(test_index(i)).name],['../data/img/',file_list(test_index(i)).name]);
            fprintf(file,[file_list(test_index(i)).name, ' ', num2str(label-1), '\r\n']);
            % I = imread([data_path, emotion_name{label}, '/', file_list(test_index(i)).name]);
            % imwrite(I,['../data/img/','ori_',file_list(test_index(i)).name]);
            % fprintf(file,['ori_', file_list(test_index(i)).name, ' ', num2str(label-1), '\r\n']);
            % imwrite(I+uint8(20*randn(size(I))),['../data/img/','nos_',file_list(test_index(i)).name]);
            % fprintf(file,['nos_', file_list(test_index(i)).name, ' ', num2str(label-1), '\r\n']);
            % imwrite(imrotate(I,15,'bilinear','crop'),['../data/img/','rtl_',file_list(test_index(i)).name]);
            % fprintf(file,['rtl_', file_list(test_index(i)).name, ' ', num2str(label-1), '\r\n']);
            % imwrite(imrotate(I,-15,'bilinear','crop'),['../data/img/','rtr_',file_list(test_index(i)).name]);
            % fprintf(file,['rtr_', file_list(test_index(i)).name, ' ', num2str(label-1), '\r\n']);
            % imwrite(flipdim(I,2),['../data/img/','flp_',file_list(test_index(i)).name]);
            % fprintf(file,['flp_', file_list(test_index(i)).name, ' ', num2str(label-1), '\r\n']);
        end
        fclose(file);
    end
end
disp('Done');
% Conver data to lmdb
% system('E:\caffe\caffe-master\Build\x64\Release\convert_imageset.exe --shuffle --resize_height=224 --resize_width=224 C:\Users\wzy\Desktop\test4\data\img\ C:\Users\wzy\Desktop\test4\data\train.txt C:\Users\wzy\Desktop\test2\data\img_train_aug_lmdb');
% system('E:\caffe\caffe-master\Build\x64\Release\convert_imageset.exe --shuffle --resize_height=224 --resize_width=224 C:\Users\wzy\Desktop\test4\data\img\ C:\Users\wzy\Desktop\test4\data\test.txt C:\Users\wzy\Desktop\test2\data\img_test_aug_lmdb');
% system('E:\caffe\caffe-master\Build\x64\Release\compute_image_mean.exe C:\Users\wzy\Desktop\test2\data\img_train_aug_lmdb C:\Users\wzy\Desktop\test2\data\face_mean_aug.binaryproto');
% /home/litp/caffe/build/tools/convert_imageset --shuffle --resize_height=224 --resize_width=224 data/img/ data/train.txt data/train_lmdb
% /home/litp/caffe/build/tools/caffe train --solver=model/solver_stage1.prototxt --weights=/home/litp/wzy/test2/model/vgg_face_caffe/VGG_FACE.caffemodel --gpu 1
% compute_image_mean data/train_lmdb data/face_mean_aug.binaryproto
% Fine-tune
% system('E:\caffe\caffe-master\Build\x64\Release\caffe.exe train --solver=C:\Users\wzy\Desktop\test1\model\solver.prototxt --weights=C:\Users\wzy\Desktop\test1\model\vgg_face_caffe\VGG_FACE.caffemodel');
% system('/home/litp/caffe/build/tools/caffe train --solver=/home/litp/wzy/test4/model/solver.prototxt --weights=/home/litp/wzy/test2/snap/mul2_iter_1000.caffemodel --gpu 1');