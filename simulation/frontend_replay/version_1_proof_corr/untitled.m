% Define the c function as an anonymous function
f = @(x) 302.2328 ./ (1 + exp(-0.0143 * (x - 101.5374)));

% Example: evaluate at a single point
x_single = 100;
y_single = f(x_single);
disp(['f(' num2str(x_single) ') = ' num2str(y_single)]);

% Example: evaluate over a range and plot
x_range = -50:50;               % vector from 0 to 200
y_range = f(x_range);           % compute function for all x
plot(x_range, y_range, 'b-', 'LineWidth', 2);
xlabel('x');
ylabel('f(x)');
title('Logistic Growth Curve');
grid on;