%--------- Add function path and set save path ----------
addpath(genpath('./data'));
addpath (genpath('./utils'));
addpath ('./requirements');
savePath = './results/';

% --------- Read instructions JSON archive -----------
json = readstruct("data/instructionsDecisionTree.json");


%% TRAINING DATASET

%--------- Load data ---------
fileName = char(json.inputDataSelection.fileName);
T_Original = readtable(['./data/' fileName]);

% -------- Remove rows with NaN --------
T_Original = rmmissing(T_Original);  % Remove rows that contain at least one NaN

% -------- Select data to use (columns with string or numerical criteria) ---------------
target_columns = json.inputDataSelection.columnCriteria.target_columns;
ignore_columns = json.inputDataSelection.columnCriteria.ignore_columns;

T_Data = selectColumns (T_Original, target_columns, ignore_columns);
T_ResultsVariable = T_Original.Death;

% --------- Categorical variable -------
catVariable = char(json.inputDataSelection.catVariable);

% --------- Identify minority and majority classes -------------
[classNames, ~, idxClass] = unique(T_ResultsVariable);
counts = accumarray(idxClass, 1);

[~, majorityIdx] = max(counts);
[~, minorityIdx] = min(counts);

majorityClass = string(classNames(majorityIdx));
minorityClass = string(classNames(minorityIdx));



%% CROSS-VALIDATION TREE
% ------------------ 1) TREE ------------------ 
CVMdl = fitctree( ...
    T_Data, T_ResultsVariable, ...
    'KFold',           5, ...
    'CategoricalPredictors', {catVariable}, ...
    'MinParentSize',   3);

% ------------------ 2) MODEL'S RESULTS ------------------------
[Label, Scores] = kfoldPredict(CVMdl);



%% CROSS-VALIDATION TREE WITH WEIGHTS
% ------------------ 1) TREE ------------------ 
WeightCVMdl = fitctreeWeightCV(T_Data, T_ResultsVariable, 5, {catVariable}, 3);

% ------------------ 2) MODEL'S RESULTS ------------------------
[wtLabel, wtScore] = kfoldPredict(WeightCVMdl);



%% CROSS-VALIDATION TREE WITH OVERSAMPLING
% ------------------ MODEL'S RESULTS ------------------------
[OSLabels, OSScores] = kfoldPredictOS (T_Data, T_ResultsVariable, 5, {catVariable}, 3);



%% TRAIN THE MODEL
% This section creates a Classification Tree model, trained with the whole
% dataset informing us of the importance of the predictor variables. It
% displays a decision tree graph and a predictors' importance bar graph and 
% table

% ------------------ 1) TREE ------------------ 
Mdl = fitctree(T_Data, T_ResultsVariable, 'CategoricalPredictors', {catVariable}, 'MinParentSize',3);

% To visualize the tree
view(Mdl,'Mode','graph'); 
savefig(fullfile(savePath, char(json.outputFileNames.decisionTree)));

% ------------------ 2) PREDICTORS' IMPORTANCE ------------------ 
imp = predictorImportance(Mdl);
predictorNames = Mdl.PredictorNames;

% To visualize the importance as a graph
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

savefig(fullfile(savePath, char(json.outputFileNames.predictorImportance)));

% To visualize the importance table
PredictorImportanceTable = table(predictorNames', imp', 'VariableNames', {'Predictor', 'Importance'});
disp('PredictorImportance_Table:');
disp(PredictorImportanceTable);



%% TEST1 - TRAINING DATASET
[test1Labels, test1Scores] = predict(Mdl, T_Data);



%% QUALITY METRICS
% === ERROR ===
    % 1) Cross-validation Sandard
        % --- Misclassification rate ---
        missClassRate = kfoldLoss(CVMdl);
        
        % --- Class-wise misclassification ---
        [missMajority, missMinority] = classwiseMisclassification(T_ResultsVariable, Label, majorityClass, minorityClass);
    
    % 2) CV Weighted
        % --- Misclassification rate ---
        missClassRateWeight  = kfoldLoss(WeightCVMdl); 
        
        % --- Class-wise misclassification ---
        [missMajorityW, missMinorityW] = classwiseMisclassification(T_ResultsVariable, wtLabel, majorityClass, minorityClass);
        
    % 3) CV Oversampled
        % --- Misclassification rate ---
        missClassRateOS = sum(~strcmp(OSLabels, T_ResultsVariable)) / numel(T_ResultsVariable);
        
        % --- Class-wise misclassification ---
        [missMajorityOS, missMinorityOS] = classwiseMisclassification(T_ResultsVariable, OSLabels, majorityClass, minorityClass);
    
    % 4) Test1 - Training Data
        % --- Misclassification rate ---
        nTotal = numel(T_ResultsVariable);
        nIncorrect = sum(~strcmp(test1Labels, T_ResultsVariable));
        
        missClassRateTest1 = nIncorrect / nTotal;
        
        % --- Class-wise misclassification ---
        [missMajorityFinal, missMinorityFinal] = classwiseMisclassification(T_ResultsVariable, test1Labels, majorityClass, minorityClass);


% === AUC ===
% Convert ground truth to binary --> Positive class is the minority class
trueBinary = string(T_ResultsVariable) == minorityClass{1};

    % 1) Cross-validation Sandard
    [Label, Score] = kfoldPredict(CVMdl);
    [~,~,~,auc1] = perfcurve(trueBinary, Score(:,2), 1);
    
    % 2) CV Weighted
    [wtLabel, wtScore] = kfoldPredict(WeightCVMdl);
    [~,~,~,auc2] = perfcurve(trueBinary, wtScore(:,2), 1);
    
    % 3) CV Oversampled
    [~,~,~,auc3] = perfcurve(trueBinary, OSScores(:,2), 1);
    
    % 4) Test1 - Training Data
    [test1Labels, test1Scores] = predict(Mdl, T_Data);
    [~,~,~,auc4] = perfcurve(trueBinary, test1Scores(:,2), 1);

% === QUALITY METRICS TABLE ===
% Prepare summary table for errors
Models = {'Standard CV'; 'Weighted CV'; 'Oversampled CV'; 'Test1 - Training Data'};
Metrics = {'OverallError'; 'MajorityError'; 'MinorityError'; 'AUC'};
OverallError = [missClassRate; missClassRateWeight; missClassRateOS; missClassRateTest1];
MajorityError = [missMajority; missMajorityW; missMajorityOS; missMajorityFinal];
MinorityError = [missMinority; missMinorityW; missMinorityOS; missMinorityFinal];
AUC = [auc1; auc2; auc3; auc4];
Matrix = [OverallError, MajorityError, MinorityError, AUC];
PlotMatrix = [OverallError'; MajorityError'; MinorityError'; AUC'];

ErrorTable = array2table(Matrix,...
    'VariableNames', Metrics,...
    'RowNames', Models);
disp('Error_Table:');
disp(ErrorTable);

PlotTable = array2table(PlotMatrix,...
    'VariableNames', Models,...
    'RowNames', Metrics);

%% CONFUSION CHART
trueLabels = string(T_ResultsVariable);
ch_Label = string(Label);
ch_wtLabel = string(wtLabel);
ch_OSLabels = string(OSLabels);
ch_trainingDatasetPredictions = string(test1Labels);


figure('Units', 'normalized', 'Position', [0.1 0.3 0.8 0.4]);
sgtitle('Confusion Matrices per model');

subplot(1,4,1);
confusionchart(trueLabels, ch_Label);
title('Standard 5-fold');

subplot(1,4,2);
confusionchart(trueLabels, ch_wtLabel);
title('Weighted 5-fold');

subplot(1,4,3);
confusionchart(trueLabels, ch_OSLabels);
title('Oversampled 5-fold');

subplot(1,4,4);
confusionchart(trueLabels, ch_trainingDatasetPredictions);
title('Final Model');

savefig(fullfile(savePath, char(json.outputFileNames.confusionCharts)));



%% ROC CURVES
figure; hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');  % Diagonal
[fp1,tp1] = perfcurve(trueBinary, Score(:,2), 1); plot(fp1,tp1, 'LineWidth', 2);
[fp2,tp2] = perfcurve(trueBinary, wtScore(:,2), 1); plot(fp2,tp2, 'LineWidth', 2);
[fp3,tp3] = perfcurve(trueBinary, OSScores(:,2), 1); plot(fp3,tp3, 'LineWidth', 2);
[fp4,tp4] = perfcurve(trueBinary, test1Scores(:,2), 1); plot(fp4,tp4, 'LineWidth', 2);

legend({ ...
    sprintf('cvStandard (AUC = %.3f)', auc1), ...
    sprintf('cvWeighted (AUC = %.3f)', auc2), ...
    sprintf('cvOversampled (AUC = %.3f)', auc3), ...
    sprintf('TrainingData (AUC = %.3f)', auc4)}, ...
    'Location', 'SouthEast');

xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC Curves for All Models');
grid on;
savefig(fullfile(savePath, char(json.outputFileNames.ROC_Curves)));


%% CREATE AN EXCEL FILE 
excelFileName = fullfile(savePath, char(json.outputFileNames.excelFileName));

% Write error summary to sheet1
writetable(ErrorTable, excelFileName, 'Sheet', 'Errors');

% Write predictor importance to sheet2
writetable(PredictorImportanceTable, excelFileName, 'Sheet', 'PredictorImportance');