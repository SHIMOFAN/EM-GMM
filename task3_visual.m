% 读取数据
filename = 'iris.csv';  
data = readtable(filename, 'ReadVariableNames', false);

% 添加列名
data.Properties.VariableNames = {'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'};

% 转换类型
data.Species = categorical(data.Species);

% 获取数值矩阵
X = data{:, 1:4};
species = data.Species;
speciesNames = categories(species);

% 设置图形样式
set(0, 'DefaultAxesFontName', 'Microsoft YaHei');  
set(0, 'DefaultTextFontName', 'Microsoft YaHei');

%% 直方图
figure('Name', '直方图', 'Position', [100 100 1200 500]);
for i = 1:4
    subplot(2, 2, i);
    histogram(X(:, i), 20, 'FaceAlpha', 0.7);
    title(data.Properties.VariableNames{i});
    xlabel('cm'); 
end

%% 双变量分析：热力图
figure('Name', '热力图');
corrMat = corr(X, 'Type', 'Pearson');
heatmap(data.Properties.VariableNames(1:4), data.Properties.VariableNames(1:4), corrMat, ...
    'Colormap', parula, 'FontSize', 12, 'CellLabelColor', 'k')

%% 多变量分析：3D散点图
figure('Name', '3D散点图');
scatter3(X(setosaIdx, 3), X(setosaIdx, 4), X(setosaIdx, 1), 15, 'r', 'filled');
hold on;
scatter3(X(versicolorIdx, 3), X(versicolorIdx, 4), X(versicolorIdx, 1), 15, 'g', 'filled');
scatter3(X(virginicaIdx, 3), X(virginicaIdx, 4), X(virginicaIdx, 1), 15, 'b', 'filled');
hold off;
xlabel('Petal length (cm)');
ylabel('Petal width (cm)');
zlabel('Sepal length (cm)');
legend(speciesNames, 'Location', 'best');
view(45, 20);  