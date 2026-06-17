clear; clc; close all;
rng(42);

%% 参数设置
pcaVarianceRatio = 90;     % PCA 保留的累计方差百分比
K = 3;                      % GMM 聚类数

%% 读取数据
fileList = {'winequality-white.csv'};

allData = [];
allQuality = [];

for i = 1:length(fileList)
    fileName = fileList{i};

    if isfile(fileName)
        T = readtable(fileName, ...
            'Delimiter', ';', ...
            'VariableNamingRule', 'preserve');

        data_i = table2array(T(:, 1:11));
        quality_i = table2array(T(:, 12));

        allData = [allData; data_i];
        allQuality = [allQuality; quality_i];

        fprintf('已读取文件：%s，样本数：%d\n', fileName, size(data_i, 1));
    else
        fprintf('未找到文件：%s\n', fileName);
    end
end

X = allData;
quality = allQuality;
fprintf('\n读取完成：共 %d 个样本，%d 个输入特征。\n', size(X,1), size(X,2));

featureNames = {
    'fixed acidity'
    'volatile acidity'
    'citric acid'
    'residual sugar'
    'chlorides'
    'free sulfur dioxide'
    'total sulfur dioxide'
    'density'
    'pH'
    'sulphates'
    'alcohol'
};

%% 标准化
X_std = zscore(X);

%% PCA 降维
[coeff, score, latent, ~, explained] = pca(X_std);
cumulativeVar = cumsum(explained);
nPC = find(cumulativeVar >= pcaVarianceRatio, 1, 'first');
fprintf('\n保留 %.0f%% 方差需要 %d 个主成分（累积方差 %.2f%%）\n', ...
    pcaVarianceRatio, nPC, cumulativeVar(nPC));

X_pca = score(:, 1:nPC);   % 降维后的数据

%% 训练 GMM
fprintf('\n使用 PCA 降维到 %d 维，K = %d 训练 GMM...\n', nPC, K);
options = statset('MaxIter', 500, 'Display', 'final');
gmmModel = fitgmdist(X_pca, K, ...
    'CovarianceType', 'full', ...
    'RegularizationValue', 1e-5, ...
    'Replicates', 10, ...
    'Options', options);
clusterIdx = cluster(gmmModel, X_pca);   % 聚类标签

%% 根据质量分数排序，为每个簇赋予质量等级标签
meanQuality = zeros(K, 1);
for k = 1:K
    meanQuality(k) = mean(quality(clusterIdx == k));
end

[~, order] = sort(meanQuality, 'ascend');

% 重新映射簇编号：1 为最低质量，K 为最高质量
newClusterIdx = zeros(size(clusterIdx));
for k = 1:K
    newClusterIdx(clusterIdx == order(k)) = k;
end
clusterIdx = newClusterIdx;

% 生成类别名称
classNames = cell(K, 1);
for k = 1:K
    if k == 1
        classNames{k} = '低质量类';
    elseif k == K
        classNames{k} = '高质量类';
    else
        classNames{k} = sprintf('中等质量类_%d', k-1);
    end
end

%% 输出聚类统计
for k = 1:K
    fprintf('%s：样本数 = %d，平均质量分数 = %.3f\n', ...
        classNames{k}, sum(clusterIdx == k), mean(quality(clusterIdx == k)));
end

%% t-SNE 可视化
Y_tsne = tsne(X_std, ...
    'NumDimensions', 2, ...
    'Perplexity', 20, ...
    'Standardize', false);
tSNE1 = Y_tsne(:,1);
tSNE2 = Y_tsne(:,2);

colors = lines(K);
figure('Color','w','Position',[100,100,800,600]);
hold on;
for k = 1:K
    scatter(tSNE1(clusterIdx==k), tSNE2(clusterIdx==k), 8, colors(k,:), 'filled', ...
        'DisplayName', classNames{k});
end
xlabel('t-SNE 1');
ylabel('t-SNE 2');
legend('Location', 'best');
hold off;
