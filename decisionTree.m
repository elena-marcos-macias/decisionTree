%--------- Add function path and set save path ----------
addpath (genpath('./utils'));
addpath ('./requirements');
savePath = './results/';

%--------- Load data ---------
fileName = char("FUS_test6.xlsx");
T_Original = readtable(['./data/' fileName]);

% -------- Remove rows with NaN --------
T_Original = rmmissing(T_Original);  % Remove rows that contain at least one NaN

% -------- Select data to use (columns with string or numerical criteria) ---------------
target_columns = [1,22];
ignore_columns = [2,3];
T_Data = selectColumns (T_Original, target_columns, ignore_columns);
T_ResultsVariable = T_Original.Death;


% ---------- Apply decision Tree model -----------
Mdl = fitctree(T_Data,T_ResultsVariable, 'CategoricalPredictors', {'Genotype'}, 'MinParentSize',3);
view(Mdl,'Mode','graph');

% --------- Importance Predictor -------
imp = predictorImportance(Mdl);

figure;
bar(imp);
title('Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = Mdl.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

% --------- Trial
%label = predict(Mdl,T_Data);

%setdiff(T_ResultsVariable, label)