%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行
%%  导入数据
res = xlsread('数据集修改2.xlsx');
%%  划分训练集和测试集
temp = 1:1:5000;

%rng('default') % 设置随机数生成器的种子以保证重现性
res = res(randperm(size(res, 1)), :);

P_train = res(temp(1: 4000), 1: 7)';
T_train = res(temp(1: 4000), 8)';
M = size(P_train, 2);

P_test = res(temp(4001: 5000), 1: 7)';
T_test = res(temp(4001: 5000), 8)';
N = size(P_test, 2);
%%  数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);


%%  数据平铺
P_train =  double(reshape(P_train, 1, 7, 1, M));
P_test  =  double(reshape(P_test , 1, 7, 1, N));

t_train = t_train';
t_test  = t_test' ;
%%  数据格式转换
P_train_array = zeros(1, 7, 1, M);
P_test_array = zeros(1, 7, 1, N);

for i = 1 : M
    P_train_array(:, :, 1, i) = P_train(:, :, 1, i);
end

for i = 1 : N
    P_test_array(:, :, 1, i)  = P_test( :, :, 1, i);
end
%%  创建模型
layers = [
    imageInputLayer([1, 7, 1])                   % 建立输入层
    convolution2dLayer([1, 3], 16, 'Padding', 'same') % 卷积层
    batchNormalizationLayer                 % 批量归一化层
    reluLayer                                % ReLU 激活层
    maxPooling2dLayer([1, 2], 'Stride', [1, 2])   % 最大池化层
    fullyConnectedLayer(64)                 % 全连接层
    dropoutLayer(0.5)                       % 增加 dropout 层的比例
    fullyConnectedLayer(1)                  % 全连接层
    regressionLayer];                       % 回归层
%% 参数设置
options = trainingOptions('adam', ...    % adam优化器
    'MiniBatchSize', 512, ...               % 修改批大小
    'MaxEpochs', 1500, ...                  % 最大迭代次数
    'InitialLearnRate', 0.001, ...          % 修改初始学习率
    'LearnRateSchedule', 'piecewise', ...   % 学习率下降
    'LearnRateDropFactor', 0.5, ...         % 修改学习率下降因子
    'LearnRateDropPeriod', 500, ...         % 经过 500 次训练后
    'Shuffle', 'every-epoch', ...           % 每次训练打乱数据集
    'Plots', 'training-progress', ...       % 画出曲线
    'Verbose', false);
%%  训练模型
net = trainNetwork(P_train_array, t_train, layers, options);

%%  仿真预测
t_sim1 = predict(net, P_train_array);
t_sim2 = predict(net, P_test_array);

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
%%  均方根误差
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);

%%  查看网络结构
analyzeNetwork(net)

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  相关指标计算
%  R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;

disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])

%  MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;

disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])

%  MBE
mbe1 = sum(T_sim1' - T_train) ./ M ;
mbe2 = sum(T_sim2' - T_test ) ./ N ;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])