clear all
close all
clc

data = readmatrix('Data_unrelevant_features_Henry_CO.xlsx');
X = data(:, 2:end-1);
y = data(:, end);

[maximum, index_maximum] = max(y);

[C, I] = mink(y, 100);
in_train = zeros(length(X), 1, 'logical');
in_train(I)=1;

all_inds = [1:500]';
mu_train = find(in_train);
mu_train_inds = [];
MU = zeros(400, 1);
mu_train_inds = mu_train;

for i=1:400
    %mei_train_inds = mei_train;
    mu_search_inds = setdiff(all_inds,mu_train_inds); 
    [dmodel,perf]=dacefit(X(mu_train_inds,:),y(mu_train_inds),@regpoly0,@corrgauss,1,1e-5,2);
    
    [mu_ypred, mu_mse] = predictor(X(mu_search_inds, :),dmodel);
    [F, G] = max(mu_mse);
    
    mu_train_inds = [mu_train_inds; mu_search_inds(G)];
    MU(i) = y(mu_train_inds(end));
    if mu_train_inds(end)==index_maximum
        break
    end
       
end

plot(1:400, MU);
