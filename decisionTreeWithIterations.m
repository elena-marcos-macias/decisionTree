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

% Create string with the results' labels and a logical vector for those results'
% labels that coincide with the % minority class, in this case 'Dead'
trueLabels = string(T_ResultsVariable);
trueBinary = trueLabels == minorityClass{1};


% Preallocate result matrices: 
errorResults = zeros(nRuns, 4, 3);  % [run, metric, model] -> metric = [Overall, Maj, Min, AUC]
confusionmatResults = zeros(2,2,nRuns,3); % [2x2 confusion matrix (actual class, predicted class), run, model]
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
    [~,~,~,auc1] = perfcurve(trueBinary, Score(:,2), 1);
    errorResults(run,:,1) = [missClassRate, missMajority, missMinority, auc1];
    cm_Label = string(Label);
    cmSt = confusionmat(trueLabels, cm_Label);
    confusionmatResults(:,:,run,1) = cmSt;

    % ----- 2. Weighted CV Tree -----
    WeightCVMdl = fitctreeWeightCV(T_Data, T_ResultsVariable, nFolds, {catVariable}, 3);
    [wtLabel, wtScore] = kfoldPredict(WeightCVMdl);
    missClassRateWeight = kfoldLoss(WeightCVMdl);
    [missMajorityW, missMinorityW] = classwiseMisclassification(T_ResultsVariable, wtLabel, majorityClass, minorityClass);
    [~,~,~,auc2] = perfcurve(trueBinary, wtScore(:,2), 1);
    errorResults(run,:,2) = [missClassRateWeight, missMajorityW, missMinorityW, auc2];
    cm_wtLabel = string(wtLabel);
    cmWt = confusionmat(trueLabels, cm_wtLabel);
    confusionmatResults(:,:,run,2) = cmWt;

    % ----- 3. Oversampled CV Tree -----
    [OSLabels, OSScores] = kfoldPredictOS(T_Data, T_ResultsVariable, nFolds, {catVariable}, 3);
    missClassRateOS = sum(~strcmp(OSLabels, T_ResultsVariable)) / numel(T_ResultsVariable);
    [missMajorityOS, missMinorityOS] = classwiseMisclassification(T_ResultsVariable, OSLabels, majorityClass, minorityClass);
    [~,~,~,auc3] = perfcurve(trueBinary, OSScores(:,2), 1);
    errorResults(run,:,3) = [missClassRateOS, missMajorityOS, missMinorityOS, auc3];
    cm_OSLabels = string(OSLabels);
    cmOS = confusionmat(trueLabels, cm_OSLabels);
    confusionmatResults(:,:,run,3) = cmOS;
end

%metrics = {'OverallError', 'MajorityError', 'MinorityError', 'AUC'};
%
%for m = 1:3
%    fprintf('\nResults for %s CV:\n', modelNames{m});
%    for k = 1:4
%        meanVal = mean(errorResults(:,k,m));
%        stdVal = std(errorResults(:,k,m));
%        fprintf('%s: Mean = %.4f, Std = %.4f\n', metrics{k}, meanVal, stdVal);
%    end
%end


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
% Obtain the models' predictions for this dataset
[test1Labels, test1Scores] = predict(Mdl, T_Data);

% Obtain the missclassification rate for all subjects in the dataset
missClassRateTest1 = sum(~strcmp(test1Labels, T_ResultsVariable)) / numel(T_ResultsVariable);

% Obtain the missclassification rates for each class separately (minority
% and majority)
[missMajorityTest1, missMinorityTest1] = classwiseMisclassification(T_ResultsVariable, test1Labels, majorityClass, minorityClass);

% Create a logical vector for those results' labels that coincide with the
% minority class, in this case 'Dead'
trueBinary = string(T_ResultsVariable) == minorityClass{1};

% Obtain the AUC
[~,~,~,aucTest1] = perfcurve(trueBinary, test1Scores(:,2), 1);



%% QUALITY METRICS TABLE
% For the 3 cross-validation models the table displays the mean value of 
% all iterations. For Test1 it displays the value of the test.

% Prepare summary table for errors
Models = {'Standard CV'; 'Weighted CV'; 'Oversampled CV'; 'Test1 - Training Data'};
Metrics = {'Mean_OverallError'; 'Std_OverallError'; 'Mean_MajorityError'; 'Std_MajorityError'; 'Mean_MinorityError'; 'Std_MinorityError'; 'Mean_AUC'; 'Std_AUC'};

means = squeeze(mean(errorResults,1))';
meanOS = means(:,1);
meanME = means(:,2);
meanmE = means(:,3);
meanAUC = means(:,4);

STDs = squeeze(std(errorResults,1))';
stdOS = STDs(:,1);
stdME = STDs(:,2);
stdmE = STDs(:,3);
stdAUC = STDs(:,4);
stdTest = ("-");

% Prepare columns
clmMeanOS = [meanOS;missClassRateTest1];
clmMeanME = [meanME;missMajorityTest1];
clmMeanmE = [meanmE;missMinorityTest1];
clmMeanAUC = [meanAUC;aucTest1];
clmStdOS = [stdOS;stdTest];
clmStdME = [stdME;stdTest];
clmStdmE = [stdmE;stdTest];
clmStdAUC = [stdAUC;stdTest];

Matrix = [clmMeanOS, clmStdOS, clmMeanME, clmStdME, clmMeanmE, clmStdmE, clmMeanAUC, clmStdAUC];
PlotMatrix = [clmMeanOS'; clmStdOS'; clmMeanME'; clmStdME'; clmMeanmE'; clmStdmE'; clmMeanAUC'; clmStdAUC'];

% Create table
ErrorTable = array2table(Matrix,...
    'VariableNames', Metrics,...
    'RowNames', Models);
disp('Error_Table:');
disp(ErrorTable);



%% CONFUSION CHART OF HISTOGRAMS - 2×8 LAYOUT
numBins = 10;

% Create one big tiledlayout (2 rows × 8 columns)
bigFig = figure('Name', 'All Confusion Matrix Histograms', 'NumberTitle', 'off');
tBig = tiledlayout(bigFig, 2, 8, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

% Define tile positions for each model's histograms
% Row 1 tiles: 1–8, Row 2 tiles: 9–16
tilePositions = { [1, 2, 9, 10], ... % Model 1
                  [3, 4, 11, 12], ... % Model 2
                  [5, 6, 13, 14] };   % Model 3
% Blank: [7, 8, 15, 16]

for m = 1:3  % loop over CV methods
    
    % --- Step 1: Determine axis limits for this model ---
    maxX = -inf;
    maxY = -inf;
    
    for pos = 1:4
        switch pos
            case 1, row = 1; col = 1;
            case 2, row = 1; col = 2;
            case 3, row = 2; col = 1;
            case 4, row = 2; col = 2;
        end
        values = squeeze(confusionmatResults(row, col, :, m));
        
        [binCounts, binEdges] = histcounts(values, numBins);
        maxX = max(maxX, max(binEdges));
        maxY = max(maxY, max(binCounts));
    end
    
    % --- Step 2: Plot histograms into assigned tiles ---
    for pos = 1:4
        ax = nexttile(tBig, tilePositions{m}(pos)); % Go to correct tile
        
        switch pos
            case 1, row = 1; col = 1;
            case 2, row = 1; col = 2;
            case 3, row = 2; col = 1;
            case 4, row = 2; col = 2;
        end
        
        values = squeeze(confusionmatResults(row, col, :, m));
        
        % Color: blue for true, red for false
        if pos == 1 || pos == 4
            faceColor = [0.2 0.6 0.8];
        else
            faceColor = [0.85 0.33 0.1];
        end
        
        histogram(ax, values, numBins, 'FaceColor', faceColor, 'EdgeColor', 'k');
        axis(ax, 'square');
        grid(ax, 'on');
        
        xlim(ax, [0, maxX]);
        ylim(ax, [0, maxY]);
        
        % Labels
        switch pos
            case 1
                ylabel(ax, majorityClass, 'FontSize', 14, 'FontWeight', 'bold');
            case 3
                ylabel(ax, minorityClass, 'FontSize', 14, 'FontWeight', 'bold');
                xlabel(ax, majorityClass, 'FontSize', 14, 'FontWeight', 'bold');
            case 4
                xlabel(ax, minorityClass, 'FontSize', 14, 'FontWeight', 'bold');
            otherwise
                xlabel(ax, '');
                ylabel(ax, '');
        end
    end
end

% --- Compute confusion matrix for test1 and display its values in tiles 7,8,15,16 ---

% Ensure test1Labels and true labels exist
if ~exist('test1Labels','var') || ~exist('T_ResultsVariable','var')
    error('Variables test1Labels or T_ResultsVariable not found in workspace.');
end

% Convert labels to cellstr for consistent display/labeling
try
    trueLabelsForCM = cellstr(T_ResultsVariable);
catch
    trueLabelsForCM = cellstr(string(T_ResultsVariable));
end
try
    predLabelsForCM = cellstr(test1Labels);
catch
    predLabelsForCM = cellstr(string(test1Labels));
end

% Compute confusion matrix (rows = true, cols = predicted)
confMatTest1 = confusionmat(trueLabelsForCM, predLabelsForCM);

% Get class names in the order used by confusionmat (stable order from true labels)
if exist('classNames','var') && numel(classNames) == size(confMatTest1,1)
    cmClassNames = classNames;
else
    cmClassNames = unique(trueLabelsForCM, 'stable');
end

[nRows, nCols] = size(confMatTest1);

% Define colors
blueColor = [0.2 0.6 0.8];
redColor  = [0.85 0.33 0.1];

if nRows == 2 && nCols == 2
    % Map 2x2 cells to individual tiles
    tileMap = [7, 8; 15, 16];   % top-left, top-right, bottom-left, bottom-right

    % Loop cells and display counts with matching label formatting & colors
    for r = 1:2
        for c = 1:2
            tileIndex = tileMap(r,c);
            axCell = nexttile(tBig, tileIndex);
            
            % Set normalized axes limits so text positions are consistent
            axis(axCell, 'square');
            axCell.XLim = [0 1];
            axCell.YLim = [0 1];
            axCell.XTick = [];
            axCell.YTick = [];
            axCell.Box = 'on';
            
            % select bg color: tiles 7 & 16 blue; 8 & 15 red
            if tileIndex == 7 || tileIndex == 16
                bgColor = blueColor;
            else
                bgColor = redColor;
            end
            axCell.Color = bgColor;
            
            % Big count in center (white for contrast)
            text(0.5, 0.60, sprintf('%d', confMatTest1(r,c)), ...
                'Parent', axCell, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'FontSize', 28, ...
                'FontWeight', 'bold', ...
                'Color', 'w');
            
            % Place class labels following the same rules as the histograms:
            lblTrue = char(cmClassNames(r));
            lblPred = char(cmClassNames(c));
            if c == 1
                % left column -> show TRUE class as ylabel (white for contrast)
                ylabel(axCell, lblTrue, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'w');
            end
            if r == 2
                % bottom row -> show PRED class as xlabel (white)
                xlabel(axCell, lblPred, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'w');
            end
        end
    end

else
    % If not 2x2, span tiles 7,8,15,16 and draw heatmap with annotations
    axConf = nexttile(tBig, 7, [2, 2]);  % spans tiles 7,8,15,16
    imagesc(axConf, confMatTest1);
    axis(axConf, 'image');
    colormap(axConf, parula);
    colorbar('peer', axConf);
    title(axConf, 'Confusion matrix (counts)', 'FontSize', 12, 'FontWeight', 'bold');

    % Annotate each cell with its numeric value
    [nr, nc] = size(confMatTest1);
    for i = 1:nr
        for j = 1:nc
            text(j, i, sprintf('%d', confMatTest1(i,j)), ...
                'Parent', axConf, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'FontSize', 11, ...
                'FontWeight', 'bold', ...
                'Color', 'w');
        end
    end

    % Use class names if available and set axis labels with same formatting
    if numel(cmClassNames) == nr
        xticks(axConf, 1:nc);
        yticks(axConf, 1:nr);
        xticklabels(axConf, cmClassNames);
        yticklabels(axConf, cmClassNames);
        xtickangle(axConf, 45);
        xlabel(axConf, 'Predicted Class', 'FontSize', 14, 'FontWeight', 'bold');
        ylabel(axConf, 'Actual Class', 'FontSize', 14, 'FontWeight', 'bold');
    else
        xlabel(axConf, 'Predicted Class', 'FontSize', 14, 'FontWeight', 'bold');
        ylabel(axConf, 'Actual Class', 'FontSize', 14, 'FontWeight', 'bold');
    end
end

% --- Global labels for the big layout ---
xlabel(tBig, 'Predicted Class', 'FontSize', 14, 'FontWeight', 'bold');
ylabel(tBig, 'Actual Class', 'FontSize', 14, 'FontWeight', 'bold');
title(tBig, 'Confusion Matrix Cell Distributions (All Models)', 'FontSize', 16);

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