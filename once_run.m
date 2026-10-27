clear all;
clc;
%rng(0);
addpath('data');


alpha = 0.1;
beta = 0.001;
gamma = 10;
lambda = 0.001;
k = 20;

% Dataset file name - modify as needed
dataset_file = 'BBCSport.mat';  % dataset file name
load(dataset_file);

% Extract base file name
[~, base_filename, ~] = fileparts(dataset_file);

% Perform corresponding preprocessing according to dataset name
fprintf('Processing dataset: %s\n', base_filename);

% Dataset preprocessing
switch base_filename
    case 'yaleA'
        X = cellfun(@transpose, X, 'UniformOutput', false);
        gt = Y;
    case 'BBCSport'
        X = cellfun(@transpose, X, 'UniformOutput', false);
        gt = Y;
    case 'MSRCV1'
        X = cellfun(@transpose, X, 'UniformOutput', false);
        gt = Y;
end

fprintf('Preprocessing of dataset %s completed\n', base_filename);

tic;
result = DCGLRC(X,alpha,beta,gamma,lambda,k,gt);
toc; 
fprintf("DCGLRC:  ACC = %5.4f, NMI = %5.4f, Purity = %5.4f, ",result(1),result(2),result(3));
fprintf("F-score = %5.4f, Precision = %5.4f, Recall = %5.4f, ",result(4),result(5),result(6));
fprintf("ARI = %5.4f \n",result(7));