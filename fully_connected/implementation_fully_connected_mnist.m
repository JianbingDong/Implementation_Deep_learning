clear all;
close all;
clc;

load mnist_uint8;

train_x = double(train_x) / 255;
test_x = double(test_x) / 255;
train_y = double(train_y);
test_y = double(test_y);

train_mean = mean(train_x);                      %训练集的均值
train_sigma = max(std(train_x), eps);            %训练集的标准差
train_x = bsxfun(@minus, train_x, train_mean);   %每个样本分别减去均值
train_x = bsxfun(@rdivide, train_x, train_sigma);%每个样本分别除以标准差

test_x = bsxfun(@minus, test_x, train_mean);
test_x = bsxfun(@rdivide, test_x, train_sigma);

architeture = [784 100 10];%设定网络结构，输入784， 隐含层100，输出10
n = numel(architeture);

train_test_flag = 'test';
 %%
%训练
if strcmp(train_test_flag, 'train') == 1
    
    %初始化权重矩阵，包括偏置bias
    weights = cell(1, n-1);
    for i = 2: n
        weights{i-1} = (rand(architeture(i), architeture(i-1) + 1) - 0.5) * 8 * sqrt(6/(architeture(i) + architeture(i-1)));
    end

    learningRate = 0.001; %学习率
    numepochs = 1;    %训练几遍
    batchsize = 50;  %每次训练50个数据

    m = size(train_x, 1);       %数据总量
    numbatches = m / batchsize; %batch组数

    figure(1);
    L = zeros(numepochs * numbatches, 1); %用于记录误差值
    ll = 1; %已训练的次数
    axis([0, size(L, 1), -0.1, 20]);
    f = plot(L);
    hold on
    for epochs = 1 : numepochs
        kk = randperm(m);%打乱数据集
        for batch = 1 : numbatches
            delete(f);
            %取一个batchsize的数据
            batch_x = train_x(kk((batch - 1) * batchsize + 1 : batch * batchsize), :);
            batch_y = train_y(kk((batch - 1) * batchsize + 1 : batch * batchsize), :);

            %正向传播
            mm = size(batch_x, 1);     %当前batch大小
            x = [ones(mm, 1) batch_x]; %在数据第一列加上1，用于偏置bias
            a{1} = x; %用于存储每一层的正向传播结果

            for ii = 2 : n-1 %计算所有隐含层
                a{ii} = tanh((a{ii - 1} * weights{ii - 1}')); %每一层的正向传播，加激活函数tanh
                a{ii} = [ones(mm,1) a{ii}];                   %添加全1列，用于偏置            
            end
%             a{n} = 1 ./ (1 + exp(-(a{n-1} * weights{n-1}')));%隐含层与输出层的正向传播，加激活函数sigmoid
            
            a{n} = exp(a{n-1} * weights{n-1}') ./ sum(exp(a{n-1} * weights{n-1}'), 2); %隐含层与输出层的正向传播，加激活函数softmax
            %计算误差
            error = -batch_y .* log(a{n}); %cross_entropy 
            error = sum(sum(error, 2)) / mm; %当前batch的平均误差
            L(ll) = error;            %记录当前batch误差

            %反向传播
            d{n-1} = (-1 .* batch_y ./ a{n}) .* (a{n} .* (1 - a{n}));
            dw{n-1} = (d{n-1}' * a{n-1}) / mm; %batch梯度均值

            for layer = n-2 : -1 : 1
                d{layer} = d{layer + 1} * weights{layer + 1};
                d{layer} = d{layer} .* (1 - (tanh(a{layer + 1})) .^ 2 );
                d{layer} = d{layer}(:, 2:end);
                dw{layer} = (d{layer}' * a{layer}) / mm;
            end

            %更新梯度
            for layer = 1:n-1
                weights{layer} = weights{layer} - learningRate * dw{layer};
            end
            title(strcat('global\_step=', num2str(ll), ', error=', num2str(error)));
            ll = ll + 1;
            f = plot(L);%打印误差
            pause(0.01);
        end

    end
    
    save model_weights weights; %保存权重
    
end


%%
%测试
if strcmp(train_test_flag, 'test') == 1
    load model_weights.mat; %载入权重
    num = randi(size(test_x, 1));
    test_pic = test_x(num, :);
    [~, label_pic] = max(test_y(num, :));
    
    test_x = [ones(1,1) test_pic];
 
    a{1} = test_x; %用于存储每一层的正向传播结果

    for ii = 2 : n-1 %计算所有隐含层
        a{ii} = tanh((a{ii - 1} * weights{ii - 1}')); %每一层的正向传播，加激活函数tanh
        a{ii} = [ones(1,1) a{ii}];                   %添加全1列，用于偏置            
    end
%     a{n} = 1 ./ (1 + exp(-(a{n-1} * weights{n-1}')));%隐含层与输出层的正向传播，加激活函数sigmoid
    a{n} = exp(a{n-1} * weights{n-1}') ./ sum(exp(a{n-1} * weights{n-1}'), 2); %隐含层与输出层的正向传播，加激活函数softmax
    [max_prob, predict] = max(a{n});%%找出预测概率最大的位置
    
    show_pic = reshape(test_pic, 28, 28);
    show_pic = show_pic';
    show_pic = imresize(show_pic, 5);
    figure;
    imshow(show_pic);
    title(['This is ', num2str(label_pic - 1), ', Predicted to be ', num2str(predict - 1)]);
end








