clc;
clear;
close all;

X1 = 1:980;
Y1 = randi([50 300],1,980);

X2 = X1;

y_logistic = 302.2328 ./ (1 + exp(-0.0143 * (X2 - 101.5374)));
y_quad = -0.0006 * X2.^2 + 0.7129 * X2 + 93.5322;

YMatrix1 = [y_logistic' y_quad'];

createfigure(X1, Y1, X2, YMatrix1);