%% 生成三个高斯分布的数据
clear; clc; close all;
rng(42);

%% 构造三个二维高斯分布
mu1 = [0, 0]; 
mu2 = [5, 5]; 
mu3 = [0, 6];

Sigma1 = [1, 0.3; 0.3, 1];
Sigma2 = [1.5, -0.4; -0.4, 1];
Sigma3 = [1, 0; 0, 1.8];

n = 500;

data1 = mvnrnd(mu1, Sigma1, n);
data2 = mvnrnd(mu2, Sigma2, n);
data3 = mvnrnd(mu3, Sigma3, n);

data = [data1; data2; data3];
true_labels = [ones(n,1); 2*ones(n,1); 3*ones(n,1)];

%% 图1：原始散点图
figure('Name', '图1 原始散点图', 'Color', 'w');
scatter(data1(:,1), data1(:,2), 4, 'red', 'filled'); hold on;
scatter(data2(:,1), data2(:,2), 4, "green", 'filled');
scatter(data3(:,1), data3(:,2), 4, 'blue', 'filled');
xlabel('x_1'); ylabel('x_2');
legend('Gaussian 1', 'Gaussian 2', 'Gaussian 3', 'Location', 'best');
axis equal; hold off;

%% 使用K-means初始化GMM
K = 3; 
N = size(data,1); 
dim = 2;
max_iter = 100;
tol = 1e-6;

[~, init_labels] = kmeans(data, K, 'MaxIter', 100, 'Replicates', 5);

pi_hat = ones(1,K) / K;
mu_hat = zeros(K,dim);
Sigma_hat = repmat(eye(dim), [1,1,K]);

for k = 1:K
    idx = init_labels == k;
    if sum(idx) > 1
        mu_hat(k,:) = mean(data(idx,:), 1);
        Sigma_tmp = cov(data(idx,:));
    else
        mu_hat(k,:) = data(randi(N), :);
        Sigma_tmp = eye(dim);
    end
    Sigma_hat(:,:,k) = ensurePositiveDefinite(Sigma_tmp, 1e-6);
end

% 记录均值迭代轨迹
mu_history = zeros(K, dim, max_iter + 1);
mu_history(:,:,1) = mu_hat;

%% EM 算法
log_likelihood = zeros(max_iter,1);

for iter = 1:max_iter
    % E step
    gamma = zeros(N, K);
    for k = 1:K
        pdf_k = mvnpdf(data, mu_hat(k,:), Sigma_hat(:,:,k));
        pdf_k(~isfinite(pdf_k)) = realmin;
        gamma(:,k) = pi_hat(k) * pdf_k;
    end
    sum_gamma = sum(gamma, 2);
    bad_rows = ~isfinite(sum_gamma) | sum_gamma <= realmin;
    sum_gamma_safe = sum_gamma;
    sum_gamma_safe(bad_rows) = realmin;
    gamma = gamma ./ sum_gamma_safe;
    
    % log-likelihood
    log_likelihood(iter) = sum(log(sum_gamma_safe));
    
    % M step
    Nk = sum(gamma, 1);
    
    % 空分量重新初始化
    if any(Nk < 1e-8) || any(~isfinite(Nk))
        fprintf('第 %d 次迭代出现空分量，重新初始化。\n', iter);
        [~, init_labels] = kmeans(data, K, 'MaxIter', 100, 'Replicates', 5);
        pi_hat = ones(1,K) / K;
        for k = 1:K
            idx = init_labels == k;
            if sum(idx) > 1
                mu_hat(k,:) = mean(data(idx,:), 1);
                Sigma_hat(:,:,k) = ensurePositiveDefinite(cov(data(idx,:)), 1e-6);
            else
                mu_hat(k,:) = data(randi(N), :);
                Sigma_hat(:,:,k) = eye(dim);
            end
        end
        mu_history(:,:,iter+1) = mu_hat;
        continue;
    end
    
    % 更新参数
    pi_hat = Nk / N;
    for k = 1:K
        mu_hat(k,:) = (gamma(:,k)' * data) / Nk(k);
    end
    for k = 1:K
        diff = data - mu_hat(k,:);
        Sigma_new = (diff' * (diff .* gamma(:,k))) / Nk(k);
        Sigma_hat(:,:,k) = ensurePositiveDefinite(Sigma_new, 1e-6);
    end
    
    mu_history(:,:,iter+1) = mu_hat;
    
    % 收敛判断
    if iter > 1 && abs(log_likelihood(iter) - log_likelihood(iter-1)) < tol
        break;
    end
end

% 截取有效迭代
actual_iter = iter;
log_likelihood = log_likelihood(1:actual_iter);
mu_history = mu_history(:,:,1:actual_iter+1);
fprintf('EM 在第 %d 步收敛, 最终对数似然为 = %.4f\n', ...
    length(log_likelihood), log_likelihood(end));

%% 使用最终参数重新计算 gamma
gamma = zeros(N, K);
for k = 1:K
    pdf_k = mvnpdf(data, mu_hat(k,:), Sigma_hat(:,:,k));
    pdf_k(~isfinite(pdf_k)) = realmin;
    gamma(:,k) = pi_hat(k) * pdf_k;
end
sum_gamma = sum(gamma, 2);
bad_rows = ~isfinite(sum_gamma) | sum_gamma <= realmin;
sum_gamma_safe = sum_gamma;
sum_gamma_safe(bad_rows) = realmin;
gamma = gamma ./ sum_gamma_safe;
gamma(~isfinite(gamma)) = 1 / K;

%% 图2：EM 收敛曲线
figure('Name', '图2 EM 收敛曲线', 'Color', 'w');
plot(1:length(log_likelihood), log_likelihood, 'b', 'LineWidth', 1);
xlabel('Iteration'); ylabel('Log-likelihood');

%% 聚类结果与精度
[~, pred_labels] = max(gamma, [], 2);
C = confusionmat(true_labels, pred_labels);
best_acc_num = 0; best_perm = 1:K;
all_perm = perms(1:K);
for i = 1:size(all_perm, 1)
    perm = all_perm(i,:);
    correct_num = sum(C(perm(1),1) + C(perm(2),2) + C(perm(3),3));
    if correct_num > best_acc_num
        best_acc_num = correct_num;
        best_perm = perm;
    end
end
pred_labels_mapped = zeros(N, 1);
for j = 1:K
    pred_labels_mapped(pred_labels == j) = best_perm(j);
end
acc = mean(pred_labels_mapped == true_labels) * 100;
fprintf('Clustering accuracy = %.2f%%\n', acc);

%% 图3：GMM 聚类结果
figure('Name', '图3 GMM 聚类结果图', 'Color', 'w');
scatter(data(:,1), data(:,2), 4, pred_labels_mapped, 'filled');
colormap([1,0,0; 0,1,0; 0,0,1]); hold on;
plot(mu_hat(:,1), mu_hat(:,2), 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y');
xlabel('x_1'); ylabel('x_2');
legend('聚类数据点', '最终均值', 'Location', 'best');
axis equal; hold off;

%% 图4：均值迭代轨迹
figure('Name', '图4 均值迭代轨迹图', 'Color', 'w');
scatter(data(:,1), data(:,2), 4, [0.75 0.75 0.75], 'filled'); hold on;
colors = lines(K);
for k = 1:K
    trajectory = squeeze(mu_history(k,:,:))';
    plot(trajectory(:,1), trajectory(:,2), '-o', 'Color', colors(k,:), 'LineWidth', 1, 'MarkerSize', 5);
    plot(trajectory(1,1), trajectory(1,2), 's', 'Color', colors(k,:), 'MarkerFaceColor', colors(k,:), 'MarkerSize', 5);
    plot(trajectory(end,1), trajectory(end,2), 'p', 'Color', colors(k,:), 'MarkerFaceColor', colors(k,:), 'MarkerSize', 9);
end
xlabel('x_1'); ylabel('x_2');
legend('数据点', '分量1轨迹', '分量1初始点', '分量1最终点', ...
       '分量2轨迹', '分量2初始点', '分量2最终点', ...
       '分量3轨迹', '分量3初始点', '分量3最终点', ...
       'Location', 'bestoutside');
axis equal; hold off;

%% 图5：最终高斯分布等高线图
figure('Name', '图5 最终高斯分布等高线图', 'Color', 'w');
scatter(data(:,1), data(:,2), 4, [0.75 0.75 0.75], 'filled'); hold on;
x_min = min(data(:,1)) - 1; x_max = max(data(:,1)) + 1;
y_min = min(data(:,2)) - 1; y_max = max(data(:,2)) + 1;
[X1, X2] = meshgrid(linspace(x_min, x_max, 150), linspace(y_min, y_max, 150));
X_grid = [X1(:), X2(:)];
for k = 1:K
    pdf_grid = mvnpdf(X_grid, mu_hat(k,:), Sigma_hat(:,:,k));
    pdf_grid = reshape(pdf_grid, size(X1));
    if all(isfinite(pdf_grid(:)))
        contour(X1, X2, pdf_grid, 5, 'LineWidth', 1.5, 'Color', colors(k,:));
    else
        fprintf('第 %d 个高斯分量的等高线数据异常，跳过绘制。\n', k);
    end
end
plot(mu_hat(:,1), mu_hat(:,2), 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y');
xlabel('x_1'); ylabel('x_2');
legend('数据点', 'Gaussian 1 等高线', 'Gaussian 2 等高线', 'Gaussian 3 等高线', '最终均值', 'Location', 'best');
axis equal; hold off;

%% 输出估计参数
disp('估计得到的混合系数 pi_hat：'); disp(pi_hat);
disp('估计得到的均值 mu_hat：'); disp(mu_hat);
disp('估计得到的协方差矩阵 Sigma_hat：');
for k = 1:K
    fprintf('第 %d 个高斯分量的协方差矩阵：\n', k);
    disp(Sigma_hat(:,:,k));
end

%% 辅助函数：强制协方差矩阵对称正定
function Sigma_pd = ensurePositiveDefinite(Sigma, epsilon)
    if isempty(Sigma) || any(isnan(Sigma(:))) || any(isinf(Sigma(:)))
        Sigma_pd = eye(2); return;
    end
    Sigma_pd = (Sigma + Sigma') / 2;
    if size(Sigma_pd,1) ~= size(Sigma_pd,2)
        Sigma_pd = eye(size(Sigma_pd,1)); return;
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