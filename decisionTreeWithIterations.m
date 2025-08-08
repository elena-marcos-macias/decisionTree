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



%% TRAIN THE DECISION TREE - TRAINING DATASET
% This section creates a Classification Tree model, trained with the whole
% dataset informing us of the importance of the predictor variables. It
% displays a decision tree graph and a predictors' importance bar graph and 
% table

% ------------------ 1) TREE ------------------ 
Mdl = fitctree(T_Data, T_ResultsVariable, 'CategoricalPredictors', {catVariable}, 'MinParentSize',3);

%To visualize the tree
view(Mdl,'Mode','graph'); 
savefig(fullfile(savePath, char(json.outputFileNames.decisionTree)));

% ------------------ 2) PREDICTORS' IMPORTANCE ------------------ 
imp = predictorImportance(Mdl);
predictorNames = Mdl.PredictorNames;

%To visualize the importance as a graph
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

%To visualize the importance table
PredictorImportanceTable = table(predictorNames', imp', 'VariableNames', {'Predictor', 'Importance'});
disp('PredictorImportance_Table:');
disp(PredictorImportanceTable);



%% CROSS-VALIDATION METHODS
% To validate the model, and since we don't have a different dataset from
% the training dataset to try it with, we are going to apply
% cross-validation methods. Since the metrics' variability is very high the
% experiment will be performed nRuns times. 1. Randomiced K-Fold method. 
% 2. Weighted K-fold to manage imbalance between minority and majority
% classes. 3. Oversampled K-fold to manage imbalance.

% Numer of folds for K-fold
nFolds = 5;

% Number of repetitions
nRuns = 1000;

% Preallocate result matrices: rows = repetitions, columns = metrics
errorResults = zeros(nRuns, 4, 3);  % [run, metric, model] -> metric = [Overall, Maj, Min, AUC]
modelNames = {'Standard', 'Weighted', 'Oversampled'};

for run = 1:nRuns
    fprintf('Running iteration %d of %d...\n', run, nRuns);

    % ----- 1. Standard CV Tree -----
    CVMdl = fitctree( ...
        T_Data, T_ResultsVariable, ...
        'KFold',           nFolds, ...
        'CategoricalPredictors', {catVariable}, ...
        'MinParentSize',   3);

    [Label, Score] = kfoldPredict(CVMdl);
    missClassRate = kfoldLoss(CVMdl);
    [missMajority, missMinority] = classwiseMisclassification(T_ResultsVariable, Label, majorityClass, minorityClass);
    trueBinary = string(T_ResultsVariable) == minorityClass{1};
    [~,~,~,auc1] = perfcurve(trueBinary, Score(:,2), 1);
    errorResults(run,:,1) = [missClassRate, missMajority, missMinority, auc1];

    % ----- 2. Weighted CV Tree -----
    WeightCVMdl = fitctreeWeightCV(T_Data, T_ResultsVariable, nFolds, {catVariable}, 3);
    [wtLabel, wtScore] = kfoldPredict(WeightCVMdl);
    missClassRateWeight = kfoldLoss(WeightCVMdl);
    [missMajorityW, missMinorityW] = classwiseMisclassification(T_ResultsVariable, wtLabel, majorityClass, minorityClass);
    [~,~,~,auc2] = perfcurve(trueBinary, wtScore(:,2), 1);
    errorResults(run,:,2) = [missClassRateWeight, missMajorityW, missMinorityW, auc2];

    % ----- 3. Oversampled CV Tree -----
    [OSLabels, OSScores] = kfoldPredictOS(T_Data, T_ResultsVariable, nFolds, {catVariable}, 3);
    missClassRateOS = sum(~strcmp(OSLabels, T_ResultsVariable)) / numel(T_ResultsVariable);
    [missMajorityOS, missMinorityOS] = classwiseMisclassification(T_ResultsVariable, OSLabels, majorityClass, minorityClass);
    [~,~,~,auc3] = perfcurve(trueBinary, OSScores(:,2), 1);
    errorResults(run,:,3) = [missClassRateOS, missMajorityOS, missMinorityOS, auc3];
end

metrics = {'OverallError', 'MajorityError', 'MinorityError', 'AUC'};

for m = 1:3
    fprintf('\nResults for %s CV:\n', modelNames{m});
    for k = 1:4
        meanVal = mean(errorResults(:,k,m));
        stdVal = std(errorResults(:,k,m));
        fprintf('%s: Mean = %.4f, Std = %.4f\n', metrics{k}, meanVal, stdVal);
    end
end


%% PLOT HISTOGRAMS PER CROSS-VALIDATION METHOD AND MEASURE
metrics = {'OverallError', 'MajorityError', 'MinorityError', 'AUC'};
modelNames = {'Standard', 'Weighted', 'Oversampled'};
colors = lines(3);
numBins = 50;

% --- 1. Find axis limits per metric across all models ---
metricLimits = struct();

for k = 1:4
    minX = inf;
    maxX = -inf;
    maxY = 0;
    
    for m = 1:3
        data = errorResults(:, k, m);
        
        % Temporary histogram to get bins and values
        htemp = histogram(data, numBins);
        
        minX = min(minX, min(htemp.BinEdges));
        maxX = max(maxX, max(htemp.BinEdges));
        maxY = max(maxY, max(htemp.Values));
        
        delete(htemp);
    end
    
    metricLimits(k).minX = minX;
    metricLimits(k).maxX = maxX;
    metricLimits(k).maxY = maxY;
end

% --- 2. Plot with metric-specific axis limits ---
figure('Name', 'All Metrics Distributions by Metric Limits', 'NumberTitle', 'off');

for m = 1:3  % rows = models
    for k = 1:4  % columns = metrics
        ax = subplot(3, 4, (m-1)*4 + k);
        data = errorResults(:, k, m);
        
        histogram(data, numBins, ...
            'FaceColor', colors(m, :), ...
            'EdgeColor', 'k');
        
        % Calculate mean and add vertical line
        meanVal = mean(data);
        hold on;
        xline(meanVal, '--r', 'LineWidth', 2);
        hold off;
        
        % Apply axis limits for this metric (column)
        xlim(ax, [metricLimits(k).minX, metricLimits(k).maxX]);
        ylim(ax, [0, metricLimits(k).maxY]);
        
        if m == 1
            title(metrics{k}, 'FontSize', 12, 'FontWeight', 'bold');
        end
        if k == 1
            ylabel(modelNames{m}, 'FontSize', 12, 'FontWeight', 'bold');
        end
        
        grid on;
    end
end

sgtitle('Distribution of Errors per Metric and Model', 'Fontsize', 16);
savefig(fullfile(savePath, char(json.outputFileNames.crossValidationHistograms)));



%% TEST1 - TRAINING DATASET
[test1Labels, test1Scores] = predict(Mdl, T_Data);



%% QUALITY METRICS TABLE
% Prepare summary table for errors
%Models = {'Standard CV'; 'Weighted CV'; 'Oversampled CV'; 'Test1 - Training Data'};
%Metrics = {'OverallError'; 'MajorityError'; 'MinorityError'; 'AUC'};
%OverallError = [missClassRate; missClassRateWeight; missClassRateOS; missClassRateTest1];
%MajorityError = [missMajority; missMajorityW; missMajorityOS; missMajorityFinal];
%MinorityError = [missMinority; missMinorityW; missMinorityOS; missMinorityFinal];
%AUC = [auc1; auc2; auc3; auc4];
%Matrix = [OverallError, MajorityError, MinorityError, AUC];
%PlotMatrix = [OverallError'; MajorityError'; MinorityError'; AUC'];
%
%ErrorTable = array2table(Matrix,...
%    'VariableNames', Metrics,...
%    'RowNames', Models);
%disp('Error_Table:');
%disp(ErrorTable);
%
%PlotTable = array2table(PlotMatrix,...
%    'VariableNames', Models,...
%    'RowNames', Metrics);