clear; clc; close all;
rng(42);

%% 读取数据
filename = 'iris.csv';
T = readtable(filename, 'ReadVariableNames', false);
T.Properties.VariableNames = {'SepalLength','SepalWidth','PetalLength','PetalWidth','Species'};

% 选取花瓣长度和宽度
X_raw = T{:, 3:4};
X = (X_raw - mean(X_raw)) ./ std(X_raw);

% 真实标签
species_str = T.Species;
true_labels = zeros(height(T),1);
true_labels(strcmp(species_str, 'Iris-setosa')) = 1;
true_labels(strcmp(species_str, 'Iris-versicolor')) = 2;
true_labels(strcmp(species_str, 'Iris-virginica')) = 3;

N = size(X,1);
dim = 2;
K = 3;
max_iter = 200;
tol = 1e-6;

% 尝试不同初始均值的EM，选取最佳对数似然
n_trials = 10;
best_loglik = -inf;
best_pi = [];
best_mu = [];
best_var = [];
best_mu_hist = [];
best_loglik_hist = [];

for trial = 1:n_trials
    % 随机选择K个样本作为初始均值
    idx_perm = randperm(N);
    mu_init = X(idx_perm(1:K), :);
    % 初始混合系数均匀
    pi_init = ones(1,K)/K;
    % 初始方差
    global_var = var(X, 1);
    var_init = repmat(global_var + 0.1, K, 1);   % 加正则化
    
    % 运行EM
    [pi_est, mu_est, var_est, mu_hist, loglik_trial] = ...
        run_em(X, K, pi_init, mu_init, var_init, max_iter, tol);
    
    if loglik_trial(end) > best_loglik
        best_loglik = loglik_trial(end);
        best_pi = pi_est;
        best_mu = mu_est;
        best_var = var_est;
        best_mu_hist = mu_hist;
        best_loglik_hist = loglik_trial;
    end
end

pi_hat = best_pi;
mu_hat = best_mu;
var_hat = best_var;
mu_history = best_mu_hist;
log_likelihood = best_loglik_hist;
actual_iter = length(log_likelihood);

%% 聚类准确率计算
gamma_final = zeros(N, K);
for k = 1:K
    pdf_k = mvnpdf_diag(X, mu_hat(k, :), var_hat(k, :));
    pdf_k(~isfinite(pdf_k)) = realmin;
    gamma_final(:, k) = pi_hat(k) * pdf_k;
end
sum_gamma = sum(gamma_final, 2);
gamma_final = gamma_final ./ max(sum_gamma, realmin);
[~, pred_labels] = max(gamma_final, [], 2);

C = confusionmat(true_labels, pred_labels);
best_acc = 0;
best_perm = 1:K;
all_perms = perms(1:K);
for i = 1:size(all_perms,1)
    perm = all_perms(i,:);
    correct = sum(C(sub2ind(size(C), perm, 1:K)));
    if correct > best_acc
        best_acc = correct;
        best_perm = perm;
    end
end
pred_labels_mapped = best_perm(pred_labels);
acc = best_acc / N * 100;
fprintf('聚类准确率 = %.2f%%\n', acc);

% 原始数据散点图
fig1 = figure('Name', '原始数据散点图', 'Color', 'w');
gscatter(X(:,1), X(:,2), true_labels, [1,0,0; 0,1,0; 0,0,1],'.', 6);
xlabel('Standardized Petal Length');
ylabel('Standardized Petal Width');
legend('Setosa', 'Versicolor', 'Virginica', 'Location', 'best');
axis equal;

% EM 对数似然收敛曲线
fig2 = figure('Name', 'EM 收敛曲线', 'Color', 'w');
plot(1:actual_iter, log_likelihood, 'b-', 'LineWidth', 1);
xlabel('Iteration');
ylabel('Log-likelihood');

% GMM 聚类结果
fig3 = figure('Name', 'GMM 聚类结果图', 'Color', 'w');
gscatter(X(:,1), X(:,2), pred_labels_mapped, [1,0,0; 0,1,0; 0,0,1],'.', 6);
hold on;
plot(mu_hat(:,1), mu_hat(:,2), 'kp', 'MarkerSize', 14, ...
     'MarkerFaceColor', 'y', 'LineWidth', 1.5);
xlabel('Standardized Petal Length');
ylabel('Standardized Petal Width');
legend('Setosa', 'Versicolor', 'Virginica', 'Final Means', 'Location', 'best');
axis equal;
hold off;

% 均值迭代轨迹图
fig4 = figure('Name', '均值迭代轨迹图', 'Color', 'w');
scatter(X(:,1), X(:,2), 6, [0.6 0.6 0.6], 'filled'); hold on;
colors = lines(K);
for k = 1:K
    traj = squeeze(mu_history(k, :, :))';
    plot(traj(:,1), traj(:,2), '-o', 'Color', colors(k,:), ...
         'LineWidth', 1.2, 'MarkerSize', 4);
    plot(traj(1,1), traj(1,2), 's', 'Color', colors(k,:), ...
         'MarkerFaceColor', colors(k,:), 'MarkerSize',4 );
    plot(traj(end,1), traj(end,2), 'p', 'Color', colors(k,:), ...
         'MarkerFaceColor', colors(k,:), 'MarkerSize', 6);
end
xlabel('Standardized Petal Length');
ylabel('Standardized Petal Width');
legend('Data points', ...
       'Comp1 trajectory', 'Comp1 start', 'Comp1 end', ...
       'Comp2 trajectory', 'Comp2 start', 'Comp2 end', ...
       'Comp3 trajectory', 'Comp3 start', 'Comp3 end', ...
       'Location', 'bestoutside');
axis equal;
hold off;

% 最终高斯分布等高线图
fig5 = figure('Name', '图5 最终高斯分布等高线图', 'Color', 'w');
scatter(X(:,1), X(:,2), 8, [0.6 0.6 0.6], 'filled'); hold on;

% 定义网格范围
x_min = min(X(:,1)) - 0.5;
x_max = max(X(:,1)) + 0.5;
y_min = min(X(:,2)) - 0.5;
y_max = max(X(:,2)) + 0.5;
[X1, X2] = meshgrid(linspace(x_min, x_max, 150), ...
                    linspace(y_min, y_max, 150));
X_grid = [X1(:), X2(:)];

for k = 1:K
    pdf_grid = mvnpdf_diag(X_grid, mu_hat(k, :), var_hat(k, :));
    pdf_grid = reshape(pdf_grid, size(X1));
    if all(isfinite(pdf_grid(:)))
        contour(X1, X2, pdf_grid, 5, 'LineWidth', 1.5, 'Color', colors(k,:));
    end
end
plot(mu_hat(:,1), mu_hat(:,2), 'kp', 'MarkerSize', 14, ...
     'MarkerFaceColor', 'y', 'LineWidth', 1.5);
xlabel('Standardized Petal Length');
ylabel('Standardized Petal Width');
legend('Data points', 'Gauss1', 'Gauss2', 'Gauss3', 'Final Means', ...
       'Location', 'best');
axis equal;
hold off;

%% 辅助函数：强制协方差矩阵对称正定
function [pi_hat, mu_hat, var_hat, mu_history, log_likelihood] = ...
    run_em(X, K, pi_init, mu_init, var_init, max_iter, tol)
    
    N = size(X,1);
    dim = size(X,2);
    pi_hat = pi_init;
    mu_hat = mu_init;
    var_hat = var_init;
    
    mu_history = zeros(K, dim, max_iter+1);
    mu_history(:,:,1) = mu_hat;
    log_likelihood = zeros(max_iter,1);
    
    for iter = 1:max_iter
        % E step
        gamma = zeros(N, K);
        for k = 1:K
            pdf_k = mvnpdf_diag(X, mu_hat(k,:), var_hat(k,:));
            pdf_k(~isfinite(pdf_k)) = realmin;
            gamma(:,k) = pi_hat(k) * pdf_k;
        end
        sum_gamma = sum(gamma,2);
        sum_gamma_safe = max(sum_gamma, realmin);
        gamma = gamma ./ sum_gamma_safe;
        log_likelihood(iter) = sum(log(sum_gamma_safe));
        
        % M step
        Nk = sum(gamma,1);
        if any(Nk < 1e-6) 
            bad = find(Nk < 1e-6);
            for b = bad
                mu_hat(b,:) = X(randi(N),:);  % 从数据中随机选一个点作为新均值
                var_hat(b,:) = var(X) + 0.1;
                pi_hat(b) = 0.1;  % 混合系数重新归一化
            end
            pi_hat = pi_hat / sum(pi_hat);
            continue;
        end
        
        pi_hat = Nk / N;
        for k = 1:K
            mu_hat(k,:) = (gamma(:,k)' * X) / Nk(k);
        end
        for k = 1:K
            diff = X - mu_hat(k,:);
            weighted_var = sum(diff.^2 .* gamma(:,k), 1) / Nk(k);
            var_hat(k,:) = weighted_var + 0.01; 
        end
        
        mu_history(:,:,iter+1) = mu_hat;
        
        if iter > 1 && abs(log_likelihood(iter) - log_likelihood(iter-1)) < tol
            break;
        end
    end
    log_likelihood = log_likelihood(1:iter);
    mu_history = mu_history(:,:,1:iter+1);
end

function pdf = mvnpdf_diag(X, mu, sigma_vec)
    d = size(X,2);
    diff = X - mu;
    inv_sigma2 = 1 ./ sigma_vec;
    mahal = sum(diff.^2 .* inv_sigma2, 2);
    det = prod(sigma_vec);
    const = 1 / sqrt((2*pi)^d * det);
    pdf = const * exp(-0.5 * mahal);
end