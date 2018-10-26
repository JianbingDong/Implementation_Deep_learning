clear all;
close all;
clc;

load mnist_uint8;

train_x = double(train_x) / 255;
test_x = double(test_x) / 255;
train_y = double(train_y);
test_y = double(test_y);

train_mean = mean(train_x);                      %ѵ�����ľ�ֵ
train_sigma = max(std(train_x), eps);            %ѵ�����ı�׼��
train_x = bsxfun(@minus, train_x, train_mean);   %ÿ�������ֱ��ȥ��ֵ
train_x = bsxfun(@rdivide, train_x, train_sigma);%ÿ�������ֱ���Ա�׼��

test_x = bsxfun(@minus, test_x, train_mean);
test_x = bsxfun(@rdivide, test_x, train_sigma);

architeture = [784 100 10];%�趨����ṹ������784�� ������100�����10
n = numel(architeture);

train_test_flag = 'test';
 %%
%ѵ��
if strcmp(train_test_flag, 'train') == 1
    
    %��ʼ��Ȩ�ؾ��󣬰���ƫ��bias
    weights = cell(1, n-1);
    for i = 2: n
        weights{i-1} = (rand(architeture(i), architeture(i-1) + 1) - 0.5) * 8 * sqrt(6/(architeture(i) + architeture(i-1)));
    end

    learningRate = 0.001; %ѧϰ��
    numepochs = 1;    %ѵ������
    batchsize = 50;  %ÿ��ѵ��50������

    m = size(train_x, 1);       %��������
    numbatches = m / batchsize; %batch����

    figure(1);
    L = zeros(numepochs * numbatches, 1); %���ڼ�¼���ֵ
    ll = 1; %��ѵ���Ĵ���
    axis([0, size(L, 1), -0.1, 20]);
    f = plot(L);
    hold on
    for epochs = 1 : numepochs
        kk = randperm(m);%�������ݼ�
        for batch = 1 : numbatches
            delete(f);
            %ȡһ��batchsize������
            batch_x = train_x(kk((batch - 1) * batchsize + 1 : batch * batchsize), :);
            batch_y = train_y(kk((batch - 1) * batchsize + 1 : batch * batchsize), :);

            %���򴫲�
            mm = size(batch_x, 1);     %��ǰbatch��С
            x = [ones(mm, 1) batch_x]; %�����ݵ�һ�м���1������ƫ��bias
            a{1} = x; %���ڴ洢ÿһ������򴫲����

            for ii = 2 : n-1 %��������������
                a{ii} = tanh((a{ii - 1} * weights{ii - 1}')); %ÿһ������򴫲����Ӽ����tanh
                a{ii} = [ones(mm,1) a{ii}];                   %���ȫ1�У�����ƫ��            
            end
%             a{n} = 1 ./ (1 + exp(-(a{n-1} * weights{n-1}')));%�����������������򴫲����Ӽ����sigmoid
            
            a{n} = exp(a{n-1} * weights{n-1}') ./ sum(exp(a{n-1} * weights{n-1}'), 2); %�����������������򴫲����Ӽ����softmax
            %�������
            error = -batch_y .* log(a{n}); %cross_entropy 
            error = sum(sum(error, 2)) / mm; %��ǰbatch��ƽ�����
            L(ll) = error;            %��¼��ǰbatch���

            %���򴫲�
            d{n-1} = (-1 .* batch_y ./ a{n}) .* (a{n} .* (1 - a{n}));
            dw{n-1} = (d{n-1}' * a{n-1}) / mm; %batch�ݶȾ�ֵ

            for layer = n-2 : -1 : 1
                d{layer} = d{layer + 1} * weights{layer + 1};
                d{layer} = d{layer} .* (1 - (tanh(a{layer + 1})) .^ 2 );
                d{layer} = d{layer}(:, 2:end);
                dw{layer} = (d{layer}' * a{layer}) / mm;
            end

            %�����ݶ�
            for layer = 1:n-1
                weights{layer} = weights{layer} - learningRate * dw{layer};
            end
            title(strcat('global\_step=', num2str(ll), ', error=', num2str(error)));
            ll = ll + 1;
            f = plot(L);%��ӡ���
            pause(0.01);
        end

    end
    
    save model_weights weights; %����Ȩ��
    
end


%%
%����
if strcmp(train_test_flag, 'test') == 1
    load model_weights.mat; %����Ȩ��
    num = randi(size(test_x, 1));
    test_pic = test_x(num, :);
    [~, label_pic] = max(test_y(num, :));
    
    test_x = [ones(1,1) test_pic];
 
    a{1} = test_x; %���ڴ洢ÿһ������򴫲����

    for ii = 2 : n-1 %��������������
        a{ii} = tanh((a{ii - 1} * weights{ii - 1}')); %ÿһ������򴫲����Ӽ����tanh
        a{ii} = [ones(1,1) a{ii}];                   %���ȫ1�У�����ƫ��            
    end
%     a{n} = 1 ./ (1 + exp(-(a{n-1} * weights{n-1}')));%�����������������򴫲����Ӽ����sigmoid
    a{n} = exp(a{n-1} * weights{n-1}') ./ sum(exp(a{n-1} * weights{n-1}'), 2); %�����������������򴫲����Ӽ����softmax
    [max_prob, predict] = max(a{n});%%�ҳ�Ԥ���������λ��
    
    show_pic = reshape(test_pic, 28, 28);
    show_pic = show_pic';
    show_pic = imresize(show_pic, 5);
    figure;
    imshow(show_pic);
    title(['This is ', num2str(label_pic - 1), ', Predicted to be ', num2str(predict - 1)]);
end








