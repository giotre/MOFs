clear all
close all
clc

data = readmatrix('Data_unrelevant_features_Working_Capacity.xlsx');
X = data(:, 2:end-1);
y = data(:, end);


[maximum, index_maximum] = max(y);

[C, I] = mink(y, 100);

in_train = zeros(length(X), 1, 'logical');
in_train(I)=1;
    
in_train = zeros(length(X), 1, 'logical');
in_train(I)=1;

all_inds = [1:500]';
mli_train = find(in_train);
mli_train_inds = [];
MLI = zeros(400, 1);
mli_train_inds = mli_train;

for i=1:400
    %mli_train_inds = mli_train;
    mli_search_inds = setdiff(all_inds,mli_train_inds); 
    [dmodel,perf]=dacefit(X(mli_train_inds,:),y(mli_train_inds),@regpoly0,@corrgauss,1,1e-5,2);
    
    [mli_ypred, mli_mse] = predictor(X(mli_search_inds, :),dmodel);
    [F, G] = max(-abs(mli_ypred-max(y(mli_train_inds)))./sqrt(mli_mse));
    
    mli_train_inds = [mli_train_inds; mli_search_inds(G)];
    MLI(i) = y(mli_train_inds(end));
    if mli_train_inds(end)==index_maximum
        break
    end
       
end

plot(1:400, MLI);
