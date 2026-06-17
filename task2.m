% 清空环境
clear; clc; close all;

%% 读取数据
filename = 'faithful.csv';
data = readtable(filename);

% 提取变量
eruptions = data.eruptions;
waiting = data.waiting;

%% 预处理
fprintf('数据行数: %d\n', height(data));
fprintf('缺失值数量: eruptions=%d, waiting=%d\n', sum(ismissing(eruptions)), sum(ismissing(waiting)));

% 基本统计量
fprintf('eruptions: 均值=%.3f, 中位数=%.3f, 标准差=%.3f\n', mean(eruptions), median(eruptions), std(eruptions));
fprintf('waiting:   均值=%.3f, 中位数=%.3f, 标准差=%.3f\n', mean(waiting), median(waiting), std(waiting));

% 标准化
eruptions_z = (eruptions - mean(eruptions)) / std(eruptions);
waiting_z = (waiting - mean(waiting)) / std(waiting);

% 散点图
figure('Position', [100, 100, 900, 400]);
subplot(1,2,1);
scatter(eruptions, waiting, 4 ,'filled', 'MarkerFaceColor', [0.2 0.4 0.8]);
xlabel('Eruptions (min)');
ylabel('Waiting (min)');
subplot(1,2,2);
scatter(eruptions_z, waiting_z, 4 , 'filled', 'MarkerFaceColor', [0.8 0.2 0.2]);
xlabel('Standardized Eruption (min)');
ylabel('Standardized Waiting (min)');

% 直方图 & 核密度估计
figure('Position', [150, 150, 1000, 400]);
subplot(1,2,1);
histogram(eruptions, 'Normalization', 'pdf', 'FaceAlpha', 0.7, 'EdgeColor', 'k');
hold on;
[f_er, xi_er] = ksdensity(eruptions);
plot(xi_er, f_er, 'r-', 'LineWidth', 1);
xlabel('Eruptions (min)');
ylabel('Probability Density');
subplot(1,2,2);
histogram(waiting, 'Normalization', 'pdf', 'FaceAlpha', 0.7, 'EdgeColor', 'k');
hold on;
[f_wt, xi_wt] = ksdensity(waiting);
plot(xi_wt, f_wt, 'r-', 'LineWidth', 1);
xlabel('Waiting (min)');
ylabel('Probability Density');

%% GMM 聚类
data_std = [eruptions_z, waiting_z];
K = 2;
N = size(data_std, 1);
dim = 2;
max_iter = 100;
tol = 1e-6;
rng(42);

%% K-means 初始化
[~, init_labels] = kmeans(data_std, K, 'MaxIter', 100, 'Replicates', 5);

pi_hat = ones(1, K) / K;
mu_hat = zeros(K, dim);
Sigma_hat = repmat(eye(dim), [1, 1, K]);

for k = 1:K
    idx = (init_labels == k);
    if sum(idx) > 1
        mu_hat(k, :) = mean(data_std(idx, :), 1);
        Sigma_tmp = cov(data_std(idx, :));
    else
        mu_hat(k, :) = data_std(randi(N), :);
        Sigma_tmp = eye(dim);
    end
    Sigma_hat(:, :, k) = ensurePositiveDefinite(Sigma_tmp, 1e-6);
end

% 记录均值迭代轨迹
mu_history = zeros(K, dim, max_iter + 1);
mu_history(:, :, 1) = mu_hat;

%% EM 算法
log_likelihood = zeros(max_iter, 1);

for iter = 1:max_iter
    % E step
    gamma = zeros(N, K);
    for k = 1:K
        pdf_k = mvnpdf(data_std, mu_hat(k, :), Sigma_hat(:, :, k));
        pdf_k(~isfinite(pdf_k)) = realmin;
        gamma(:, k) = pi_hat(k) * pdf_k;
    end

    sum_gamma = sum(gamma, 2);
    bad_rows = ~isfinite(sum_gamma) | sum_gamma <= realmin;
    sum_gamma_safe = sum_gamma;
    sum_gamma_safe(bad_rows) = realmin;
    gamma = gamma ./ sum_gamma_safe;
    gamma(bad_rows, :) = 1 / K;

    % log-likelihood
    log_likelihood(iter) = sum(log(sum_gamma_safe));

    % M step
    Nk = sum(gamma, 1);
    if any(Nk < 1e-8) || any(~isfinite(Nk))
        fprintf('第 %d 次迭代出现空分量，重新初始化。\n', iter);
        [~, init_labels] = kmeans(data_std, K, 'MaxIter', 100, 'Replicates', 5);
        pi_hat = ones(1, K) / K;
        for k = 1:K
            idx = (init_labels == k);
            if sum(idx) > 1
                mu_hat(k, :) = mean(data_std(idx, :), 1);
                Sigma_hat(:, :, k) = ensurePositiveDefinite(cov(data_std(idx, :)), 1e-6);
            else
                mu_hat(k, :) = data_std(randi(N), :);
                Sigma_hat(:, :, k) = eye(dim);
            end
        end
        mu_history(:, :, iter+1) = mu_hat;
        continue;
    end

    % 更新混合系数
    pi_hat = Nk / N;

    % 更新均值
    for k = 1:K
        mu_hat(k, :) = (gamma(:, k)' * data_std) / Nk(k);
    end

    % 更新协方差矩阵
    for k = 1:K
        diff = data_std - mu_hat(k, :);
        Sigma_new = (diff' * (diff .* gamma(:, k))) / Nk(k);
        Sigma_hat(:, :, k) = ensurePositiveDefinite(Sigma_new, 1e-6);
    end

    mu_history(:, :, iter+1) = mu_hat;

    % 收敛判断
    if iter > 1 && abs(log_likelihood(iter) - log_likelihood(iter-1)) < tol
        break;
    end
end

actual_iter = iter;
log_likelihood = log_likelihood(1:actual_iter);
mu_history = mu_history(:, :, 1:actual_iter+1);

fprintf('EM 在第 %d 步收敛, 最终对数似然为 = %.4f\n', actual_iter, log_likelihood(end));

%% 使用最终参数重新计算gamma
gamma = zeros(N, K);
for k = 1:K
    pdf_k = mvnpdf(data_std, mu_hat(k, :), Sigma_hat(:, :, k));
    pdf_k(~isfinite(pdf_k)) = realmin;
    gamma(:, k) = pi_hat(k) * pdf_k;
end
sum_gamma = sum(gamma, 2);
bad_rows = ~isfinite(sum_gamma) | sum_gamma <= realmin;
sum_gamma_safe = sum_gamma;
sum_gamma_safe(bad_rows) = realmin;
gamma = gamma ./ sum_gamma_safe;
gamma(bad_rows, :) = 1 / K;

[~, pred_labels] = max(gamma, [], 2);

%% EM 收敛曲线
figure('Name', 'EM收敛曲线', 'Color', 'w');
plot(1:length(log_likelihood), log_likelihood, 'b-', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Log-likelihood');
grid on;

%% GMM 聚类结果
figure('Name', 'GMM聚类结果', 'Color', 'w');
scatter(data_std(:,1), data_std(:,2), 6, pred_labels, 'filled');
colormap(lines(K));
hold on;
plot(mu_hat(:,1), mu_hat(:,2), 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y', 'LineWidth', 1.5);
xlabel('Standardized Eruptions');
ylabel('Standardized Waiting');
legend('聚类数据点', '最终均值', 'Location', 'best');
axis equal;
hold off;

%% 均值迭代轨迹图
figure('Name', '均值迭代轨迹', 'Color', 'w');
scatter(data_std(:,1), data_std(:,2), 8, [0.75 0.75 0.75], 'filled');
hold on;
colors = lines(K);
for k = 1:K
    trajectory = squeeze(mu_history(k, :, :))';
    plot(trajectory(:,1), trajectory(:,2), '-o', 'Color', colors(k,:), 'LineWidth', 1, 'MarkerSize', 5);
    plot(trajectory(1,1), trajectory(1,2), 's', 'Color', colors(k,:), 'MarkerFaceColor', colors(k,:), 'MarkerSize', 8);
    plot(trajectory(end,1), trajectory(end,2), 'p', 'Color', colors(k,:), 'MarkerFaceColor', colors(k,:), 'MarkerSize', 10);
end
xlabel('Standardized Eruptions');
ylabel('Standardized Waiting');
legend_entries = {'数据点'};
for k = 1:K
    legend_entries{end+1} = sprintf('分量%d轨迹', k);
    legend_entries{end+1} = sprintf('分量%d初始点', k);
    legend_entries{end+1} = sprintf('分量%d最终点', k);
end
legend(legend_entries, 'Location', 'bestoutside');
axis equal;
hold off;

%% 最终高斯分布等高线图
figure('Name', '等高线图', 'Color', 'w');
scatter(data_std(:,1), data_std(:,2), 8, [0.75 0.75 0.75], 'filled');
hold on;

x_min = min(data_std(:,1)) - 0.5;
x_max = max(data_std(:,1)) + 0.5;
y_min = min(data_std(:,2)) - 0.5;
y_max = max(data_std(:,2)) + 0.5;
[X1, X2] = meshgrid(linspace(x_min, x_max, 150), linspace(y_min, y_max, 150));
X_grid = [X1(:), X2(:)];

for k = 1:K
    pdf_grid = mvnpdf(X_grid, mu_hat(k, :), Sigma_hat(:, :, k));
    pdf_grid = reshape(pdf_grid, size(X1));
    if all(isfinite(pdf_grid(:)))
        contour(X1, X2, pdf_grid, 5, 'LineWidth', 1.5, 'Color', colors(k, :));
    else
        fprintf('第 %d 个高斯分量的等高线数据异常，跳过绘制。\n', k);
    end
end

plot(mu_hat(:,1), mu_hat(:,2), 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y', 'LineWidth', 1.5);
xlabel('Standardized Eruptions');
ylabel('Standardized Waiting');
legend_contour = {'数据点'};
for k = 1:K
    legend_contour{end+1} = sprintf('分量%d 等高线', k);
end
legend_contour{end+1} = '最终均值';
legend(legend_contour, 'Location', 'best');
axis equal;
hold off;

%% 计算并打印源数据的二维均值和协方差
data_orig = [eruptions, waiting];
mu_sample = mean(data_orig);
Sigma_sample = cov(data_orig);

fprintf('原始数据的二维均值和协方差\n');
fprintf('样本均值向量:\n');
disp(mu_sample);
fprintf('样本协方差矩阵:\n');
disp(Sigma_sample);

%% 将参数反标准化
mean_er = mean(eruptions); std_er = std(eruptions);
mean_wt = mean(waiting);   std_wt = std(waiting);
std_vec = [std_er, std_wt];
mean_vec = [mean_er, mean_wt];

mu_orig = mu_hat .* std_vec + mean_vec;
Sigma_orig = zeros(2, 2, K);
for k = 1:K
    Sigma_orig(:, :, k) = diag(std_vec) * Sigma_hat(:, :, k) * diag(std_vec);
end

fprintf('原始数据空间的 GMM 参数估计\n');
fprintf('估计的均值 (原始尺度):\n');
disp(mu_orig);
fprintf('估计的协方差矩阵 (原始尺度):\n');
for k = 1:K
    fprintf('第 %d 个高斯分量的协方差矩阵:\n', k);
    disp(Sigma_orig(:, :, k));
end

%% 辅助函数：强制协方差矩阵对称正定
function Sigma_pd = ensurePositiveDefinite(Sigma, epsilon)
    if isempty(Sigma) || any(isnan(Sigma(:))) || any(isinf(Sigma(:)))
        Sigma_pd = eye(2);
        return;
    end
    Sigma_pd = (Sigma + Sigma') / 2;
    if size(Sigma_pd,1) ~= size(Sigma_pd,2)
        Sigma_pd = eye(size(Sigma_pd,1));
        return;
    end
    d = size(Sigma_pd, 1);
    jitter = epsilon;
    max_try = 20;
    for i = 1:max_try
        [~, p] = chol(Sigma_pd);
        if p == 0
            Sigma_pd = (Sigma_pd + Sigma_pd') / 2;
            return;
        else
            Sigma_pd = Sigma_pd + jitter * eye(d);
            jitter = jitter * 10;
        end
    end
    [V, D] = eig(Sigma_pd);
    eig_values = diag(D);
    eig_values(eig_values < epsilon) = epsilon;
    Sigma_pd = V * diag(eig_values) * V';
    Sigma_pd = real((Sigma_pd + Sigma_pd') / 2);
    Sigma_pd = Sigma_pd + epsilon * eye(d);
end