clear all
close all
clc

data = readmatrix('Data_relevant_features_Henry_CO2.xlsx');
X = data(:, 2:end-1);
y = data(:, end);
[maximum, index_maximum] = max(y);

[C, I] = mink(y, 100);

in_train = zeros(length(X), 1, 'logical');
in_train(I)=1;

all_inds = [1:500]';
mei_train = find(in_train);
mei_train_inds = [];
MEI = zeros(400, 1);
mei_train_inds = mei_train;


for i=1:400
    %mei_train_inds = mei_train;
    mei_search_inds = setdiff(all_inds,mei_train_inds); 
    [dmodel,perf]=dacefit(X(mei_train_inds,:),y(mei_train_inds),@regpoly0,@corrgauss,1,1e-5,2);
    
    [mei_ypred, dy, mei_mse] = predictor(X(mei_search_inds, :),dmodel);
    [F, G] = max(mei_ypred);
    
    mei_train_inds = [mei_train_inds; mei_search_inds(G)];
    MEI(i) = y(mei_train_inds(end));
    if mei_train_inds(end)==index_maximum
        break
    end
       
end

plot(1:400, MEI);
