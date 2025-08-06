%--------- Add function path and set save path ----------
addpath (genpath('./utils'));
addpath ('./requirements');
savePath = './results/';


%% TRAINING DATASET

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

%% Identify minority and majority classes
[classNames, ~, idxClass] = unique(T_ResultsVariable);
counts = accumarray(idxClass, 1);

[~, majorityIdx] = max(counts);
[~, minorityIdx] = min(counts);

majorityClass = classNames(majorityIdx);
minorityClass = classNames(minorityIdx);


%% CROSS-VALIDATION TREE

% ------------------ 1) TREE ------------------ 
CVMdl = fitctree( ...
    T_Data, T_ResultsVariable, ...
    'KFold',           5, ...
    'CategoricalPredictors', {'Genotype'}, ...
    'MinParentSize',   3);

% ------------------ 2) MODEL'S RESULTS ------------------------
Label = kfoldPredict(CVMdl);

% --- Misclassification rate ---
missClassRate = kfoldLoss(CVMdl);

% --- Class-wise misclassification ---
[missMajority, missMinority] = classwiseMisclassification(T_ResultsVariable, Label, majorityClass, minorityClass);


%% CROSS-VALIDATION TREE WITH WEIGHTS

WeightCVMdl = fitctreeWeightCV(T_Data, T_ResultsVariable, 5, {'Genotype'}, 3);

% ------------------ 3) MODEL'S RESULTS ------------------------
[wtLabel, wtScore] = kfoldPredict(WeightCVMdl);

% --- Misclassification rate ---
missClassRateWeight  = kfoldLoss(WeightCVMdl); 

% --- Class-wise misclassification ---
[missMajorityW, missMinorityW] = classwiseMisclassification(T_ResultsVariable, wtLabel, majorityClass, minorityClass);


%% CROSS-VALIDATION TREE WITH OVERSAMPLING

% ------------------ MODEL'S RESULTS ------------------------
[OSLabels, OSScores] = kfoldPredictOS (T_Data, T_ResultsVariable, 5, {'Genotype'}, 3);

% --- Misclassification rate ---
missClassRateOS = sum(~strcmp(OSLabels, T_ResultsVariable)) / numel(T_ResultsVariable);

% --- Class-wise misclassification ---
[missMajorityOS, missMinorityOS] = classwiseMisclassification(T_ResultsVariable, OSLabels, majorityClass, minorityClass);


%% TRAIN THE MODEL

Mdl = fitctree(T_Data, T_ResultsVariable, 'CategoricalPredictors', {'Genotype'}, 'MinParentSize',3);

% VIEW THE TREE
view(Mdl,'Mode','graph');
savefig(fullfile(savePath, 'DecisionTree.fig'));

% PREDICTORS' IMPORTANCE
imp = predictorImportance(Mdl);
predictorNames = Mdl.PredictorNames;

figure;
bar(imp);
title('Predictor Importance Estimates');
ylabel('Importance');
xlabel('Predictors');

ax = gca;
ax.XTick = 1:numel(predictorNames);
ax.XTickLabel = predictorNames;
ax.XTickLabelRotation = 45;
ax.TickLabelInterpreter = 'none';

savefig(fullfile(savePath, 'PredictorImportance.fig'));

%% TEST DATASET

DeathFUS = predict(Mdl,T_Data);

% ------------------ Evaluate model ------------------------
nTotal = numel(T_ResultsVariable);
nIncorrect = sum(~strcmp(DeathFUS, T_ResultsVariable));

testError = nIncorrect / nTotal;

% --- Class-wise misclassification ---
[missMajorityFinal, missMinorityFinal] = classwiseMisclassification(T_ResultsVariable, DeathFUS, majorityClass, minorityClass);


%% CONFUSION CHART

trueLabels = string(T_ResultsVariable);
ch_Label = string(Label);
ch_wtLabel = string(wtLabel);
ch_OSLabels = string(OSLabels);
ch_trainingDatasetPredictions = string(DeathFUS);

figure('Units', 'normalized', 'Position', [0.1 0.3 0.8 0.4]);
set(gca,'DataAspectRatio',[6 .5 0.2]);
sgtitle ('Confusion Matrices per model');

subplot(1,4,1);
confusionchart(trueLabels, ch_Label);
title('Standard 5-fold cross-validation');

subplot(1,4,2);
confusionchart(trueLabels, ch_wtLabel);
title('Weighted 5-f cv');

subplot(1,4,3);
confusionchart(trueLabels, ch_OSLabels);
title('Oversampled 5-f cv');

subplot(1,4,4);
confusionchart(trueLabels, ch_trainingDatasetPredictions);
title('Final Model (with Training Data)');

savefig(fullfile(savePath, 'ConfusionCharts.fig'));


%% ROC CURVES AND AUC COMPUTATION

% === Convert ground truth to binary ===
% Positive class is the minority class
trueBinary = strcmp(T_ResultsVariable, minorityClass{1});

% === Get score outputs for each model ===

% 1. Standard CV
[Label, Score] = kfoldPredict(CVMdl);
[~,~,~,auc1] = perfcurve(trueBinary, Score(:,2), 1);

% 2. Weighted CV
[wtLabel, wtScore] = kfoldPredict(WeightCVMdl);
[~,~,~,auc2] = perfcurve(trueBinary, wtScore(:,2), 1);

% 3. Oversampled CV

[~,~,~,auc3] = perfcurve(trueBinary, OSScores(:,2), 1);


% 4. Final model (training data)
[DeathFUS, finalScores] = predict(Mdl, T_Data);
[~,~,~,auc4] = perfcurve(trueBinary, finalScores(:,2), 1);

% === Plot all ROC curves ===
figure; hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1.2);  % Diagonal
[fp1,tp1] = perfcurve(trueBinary, Score(:,2), 1); plot(fp1,tp1, 'LineWidth', 2);
[fp2,tp2] = perfcurve(trueBinary, wtScore(:,2), 1); plot(fp2,tp2, 'LineWidth', 2);
[fp3,tp3] = perfcurve(trueBinary, OSScores(:,2), 1); plot(fp3,tp3, 'LineWidth', 2);
[fp4,tp4] = perfcurve(trueBinary, finalScores(:,2), 1); plot(fp4,tp4, 'LineWidth', 2);

legend({ ...
    sprintf('Standard (AUC = %.3f)', auc1), ...
    sprintf('Weighted (AUC = %.3f)', auc2), ...
    sprintf('Oversampled (AUC = %.3f)', auc3), ...
    sprintf('Final (AUC = %.3f)', auc4)}, ...
    'Location', 'SouthEast');

xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC Curves for All Models');
grid on;
savefig(fullfile(savePath, 'ROC_Curves.fig'));


%% CREATE AN EXCEL FILE 

% Prepare summary table for errors

Models = {'Standard CV'; 'Weighted CV'; 'Oversampled CV'; 'Final Model'};
OverallError = [missClassRate; missClassRateWeight; missClassRateOS; testError];
MajorityError = [missMajority; missMajorityW; missMajorityOS; missMajorityFinal];
MinorityError = [missMinority; missMinorityW; missMinorityOS; missMinorityFinal];
AUC = [auc1; auc2; auc3; auc4];

ErrorSummary = table(Models, OverallError, MajorityError, MinorityError, AUC);
fprintf('Summary_Table: %.3f\n', ErrorSummary);

% Prepare predictor importance table
PredictorImportanceTable = table(predictorNames', imp', 'VariableNames', {'Predictor', 'Importance'});
fprintf('PredictorImportance_Table: %.3f\n', ErrorSummary);

% Write to Excel
excelFileName = fullfile(savePath, 'Model_Errors_and_Importance.xlsx');

% Write error summary to sheet1
writetable(ErrorSummary, excelFileName, 'Sheet', 'Errors');

% Write predictor importance to sheet2
writetable(PredictorImportanceTable, excelFileName, 'Sheet', 'PredictorImportance');