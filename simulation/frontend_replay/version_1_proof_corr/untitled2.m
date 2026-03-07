%% MATLAB script: Plot vehicle growth and fitted models
% This script reads the vehicle counts from 'growth.txt', defines the
% logistic and quadratic equations with the coefficients from your analysis,
% and generates a plot with the raw data and fitted curves.

clear; close all; clc;

%% 1. Read vehicle counts from growth.txt
filename = 'growth.txt';
fid = fopen(filename, 'r');
if fid == -1
    error('Cannot open %s. Make sure the file is in the current directory.', filename);
end

% Locate the line containing the vehicle counts
data_line = '';
while ~feof(fid)
    line = fgetl(fid);
    if contains(line, 'Vehicle counts per frame:')
        data_line = line;
        break;
    end
end
fclose(fid);

if isempty(data_line)
    error('Could not find the line with vehicle counts in %s', filename);
end

% Extract the numbers between square brackets
tokens = regexp(data_line, '\[(.*?)\]', 'tokens');
if isempty(tokens)
    error('No numbers found in the vehicle counts line.');
end
num_str = tokens{1}{1};

% Convert the comma‑separated string to a numeric array
counts = str2num(num_str);   %#ok<ST2NM>
fprintf('Loaded %d frames.\n', length(counts));

% Frame indices (starting at 0)
x = 0:length(counts)-1;

%% 2. Define the fitted equations with high‑precision coefficients
% Logistic: y = L / (1 + exp(-k*(x - x0)))
L = 302.2328;
k = 0.0143;
x0 = 101.5374;
logistic_fit = @(x) L ./ (1 + exp(-k * (x - x0)));

% Quadratic polynomial: y = a*x^2 + b*x + c
a = -0.0005518667521387619;
b =  0.7129391105869533;
c = 93.5322316774016;
quadratic_fit = @(x) a*x.^2 + b*x + c;

%% 3. Create a smooth x‑axis for plotting the fitted curves
x_fit = linspace(min(x), max(x), 1000);

%% 4. Plot the data and the models
figure('Position', [100 100 800 500]);
plot(x, counts, 'b-', 'LineWidth', 1.2, 'DisplayName', 'Observed counts');
hold on;
plot(x_fit, logistic_fit(x_fit), 'r--', 'LineWidth', 2, ...
     'DisplayName', sprintf('Logistic: y = %.4f / (1 + e^{-%.4f (x-%.4f)})', L, k, x0));
plot(x_fit, quadratic_fit(x_fit), 'g-.', 'LineWidth', 2, ...
     'DisplayName', sprintf('Quadratic: y = %.4f x^2 + %.4f x + %.4f', a, b, c));
xlabel('Frame number');
ylabel('Number of vehicles');
title('Vehicle Growth and Fitted Models');
legend('Location', 'best');
grid on;
hold off;

%% 5. Compute R² values (using the same x points as the data)
y_logistic = logistic_fit(x);
y_quadratic = quadratic_fit(x);

SS_res_log = sum((counts - y_logistic).^2);
SS_res_quad = sum((counts - y_quadratic).^2);
SS_tot = sum((counts - mean(counts)).^2);

R2_log = 1 - SS_res_log / SS_tot;
R2_quad = 1 - SS_res_quad / SS_tot;

fprintf('\n--- Goodness of fit (computed from the data) ---\n');
fprintf('Logistic R²  : %.4f\n', R2_log);
fprintf('Quadratic R² : %.4f\n', R2_quad);

% Note: Small differences from the values in compare.txt may arise because
% the coefficients used here are rounded to the digits shown in your files.