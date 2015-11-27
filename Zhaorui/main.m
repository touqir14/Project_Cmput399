clear all; close all; clc;
addpath('./data');
addpath('./DeepLearnToolBox');
load mnist_uint8;

k1=60000;
train_x=train_x([1:k1],:);
train_y=train_y([1:k1],:);
k2=10000;
test_x=test_x([1:k2],:);
test_y=test_y([1:k2],:);

% treat the samples as images, so convert them into 28*28
train_x = double(reshape(train_x',28,28,k1))/255;
test_x = double(reshape(test_x',28,28,k2))/255;
train_y = double(train_y');
test_y = double(test_y');

cnn.layers = {
    struct('type', 'i')
    struct('type', 'c', 'outputmaps', 3, 'kernelsize',5)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps',6, 'kernelsize',5)
    struct('type', 's', 'scale', 2)
};

cnn = cnnsetup(cnn, train_x, train_y);

prompt = {'Learning rate','batchsize','# Epchos'};
dialog_title = 'GUI';
num_lines = 1;
default = {'0.1','10','20'};
[user_input] = inputdlg(prompt,dialog_title,num_lines,default);
 
opts.alpha = str2num(user_input{1});
opts.batchsize =str2num(user_input{2}); 
opts.numepochs = str2num(user_input{3});

cnn = cnntrain(cnn, train_x, train_y, opts);

[net,er, bad, h, a] = cnntest(cnn, test_x, test_y);


for i=1:2 %仅仅显示前500的验证情况
    figure; %每张图100个
    scrsz = get(0,'ScreenSize');
    set(gcf,'Position',scrsz);
    s=100*(i-1);
    for r=1:100
        p=s+r;
        classify_numbers=h(p)-1;
        I=test_x(:,:,p);
        subplot(10,10,r);imshow(I);title(num2str(classify_numbers));
    end
end

for i=1:10
numsum(i)=numel(find(a==i));
numRightsum(i)=numel(find(a==h&a==i));
end
figure;
 bar( numRightsum./numsum *100)
 set(gca, 'XTick', [1:1:10]);
 set(gca,'XTickLabel',{'0','1','2','3','4','5','6 ','7','8','9 '});
 xlabel('Numbers');
 ylabel('Accuracy rate%');
 title(strcat('Accuracy rate for each number(Total:',num2str((1-er)*100),')'));
 grid on

figure;
plot(cnn.rL);% convergence of loss in each iteration 
disp([num2str((1-er)*100) '% accuracy rate']);
