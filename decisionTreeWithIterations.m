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
CoordBestAUC = zeros(3, 2, 3); % [points to plot, axes, model] -> axes = [X, Y]
    CurrentBestauc = zeros(1,3); % [auc value, model] 
CoodWorstAUC = zeros(3, 2, 3); % [points to plot, axes, model] -> axes = [X, Y]
    CurrentWorstauc = zeros (1,3); % [auc value, model] 
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
    [fp1,tp1,~,auc1] = perfcurve(trueBinary, Score(:,2), 1);
    errorResults(run,:,1) = [missClassRate, missMajority, missMinority, auc1];
    cm_Label = string(Label);
    cmSt = confusionmat(trueLabels, cm_Label);
    confusionmatResults(:,:,run,1) = cmSt;
    if CurrentBestauc(1) < auc1
        CurrentBestauc(1) = auc1;
        CoordBestAUC (:,:,1) = [fp1, tp1];
    end
    if CurrentWorstauc(1) < auc1
        CurrentWorstauc(1) = auc1;
        CoordWorstAUC (:,:,1) = [fp1, tp1];
    end

    % ----- 2. Weighted CV Tree -----
    WeightCVMdl = fitctreeWeightCV(T_Data, T_ResultsVariable, nFolds, {catVariable}, 3);
    [wtLabel, wtScore] = kfoldPredict(WeightCVMdl);
    missClassRateWeight = kfoldLoss(WeightCVMdl);
    [missMajorityW, missMinorityW] = classwiseMisclassification(T_ResultsVariable, wtLabel, majorityClass, minorityClass);
    [fp2,tp2,~,auc2] = perfcurve(trueBinary, wtScore(:,2), 1);
    errorResults(run,:,2) = [missClassRateWeight, missMajorityW, missMinorityW, auc2];
    cm_wtLabel = string(wtLabel);
    cmWt = confusionmat(trueLabels, cm_wtLabel);
    confusionmatResults(:,:,run,2) = cmWt;
    if CurrentBestauc(2) < auc2
        CurrentBestauc(2) = auc2;
        CoordBestAUC (:,:,2) = [fp2, tp2];
    end
    if CurrentWorstauc(2) < auc2
        CurrentWorstauc(2) = auc2;
        CoordWorstAUC (:,:,2) = [fp2, tp2];
    end

    % ----- 3. Oversampled CV Tree -----
    [OSLabels, OSScores] = kfoldPredictOS(T_Data, T_ResultsVariable, nFolds, {catVariable}, 3);
    missClassRateOS = sum(~strcmp(OSLabels, T_ResultsVariable)) / numel(T_ResultsVariable);
    [missMajorityOS, missMinorityOS] = classwiseMisclassification(T_ResultsVariable, OSLabels, majorityClass, minorityClass);
    [fp3,tp3,~,auc3] = perfcurve(trueBinary, OSScores(:,2), 1);
    errorResults(run,:,3) = [missClassRateOS, missMajorityOS, missMinorityOS, auc3];
    cm_OSLabels = string(OSLabels);
    cmOS = confusionmat(trueLabels, cm_OSLabels);
    confusionmatResults(:,:,run,3) = cmOS;
    if CurrentBestauc(3) < auc3
        CurrentBestauc(3) = auc3;
        CoordBestAUC (:,:,3) = [fp3, tp3];
    end
    if CurrentWorstauc(3) < auc3
        CurrentWorstauc(3) = auc3;
        CoordWorstAUC (:,:,3) = [fp3, tp3];
    end


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
[fp4,tp4,~,aucTest1] = perfcurve(trueBinary, test1Scores(:,2), 1);



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
bigFig = figure('Name', 'All Confusion Matrix Histograms', 'NumberTitle', 'off',...
    'Position', [65,248.2,1415.2,353.6]);
tBig = tiledlayout(bigFig, 2, 8, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

% Define tile positions for each model's histograms (keeps your original layout)
tilePositions = { [1, 2, 9, 10], ... % Model 1
                  [3, 4, 11, 12], ... % Model 2
                  [5, 6, 13, 14] };   % Model 3
% Blank (for confusion counts/heatmap): [7, 8, 15, 16]

% --- Paso 0: Calcular límites globales ---
maxXGlobal = -inf;
maxYGlobal = -inf;
for m = 1:3
    for pos = 1:4
        switch pos
            case 1, row = 1; col = 1;
            case 2, row = 1; col = 2;
            case 3, row = 2; col = 1;
            case 4, row = 2; col = 2;
        end
        values = squeeze(confusionmatResults(row, col, :, m));
        [binCounts, binEdges] = histcounts(values, numBins);
        if ~isempty(binEdges)
            maxXGlobal = max(maxXGlobal, max(binEdges));
        end
        if ~isempty(binCounts)
            maxYGlobal = max(maxYGlobal, max(binCounts));
        end
    end
end

% Safety fallback if histcounts returned empty for any reason
if ~isfinite(maxXGlobal) || maxXGlobal == -inf
    maxXGlobal = 1;
end
if ~isfinite(maxYGlobal) || maxYGlobal == -inf
    maxYGlobal = 1;
end

% --- Paso 1: Dibujar histogramas usando límites globales ---
% Keep an axes handle for each of the 16 tiles so we can compute bounding boxes later
axesHandles = gobjects(16,1);

for m = 1:3
    for pos = 1:4
        tileIdx = tilePositions{m}(pos);
        ax = nexttile(tBig, tileIdx);
        axesHandles(tileIdx) = ax;
        
        switch pos
            case 1, row = 1; col = 1;
            case 2, row = 1; col = 2;
            case 3, row = 2; col = 1;
            case 4, row = 2; col = 2;
        end
        
        values = squeeze(confusionmatResults(row, col, :, m));
        
        % Color: blue for true (diagonal), red for false (off-diagonal)
        if pos == 1 || pos == 4
            faceColor = [0.2 0.6 0.8];
        else
            faceColor = [0.85 0.33 0.1];
        end
        
        histogram(ax, values, numBins, 'FaceColor', faceColor, 'EdgeColor', 'k');
        axis(ax, 'square');
        grid(ax, 'on');
        
        xlim(ax, [0, maxXGlobal]);
        ylim(ax, [0, maxYGlobal]);
        
        % Labels (same formatting as original)
        switch pos
            case 1
                ylabel(ax, majorityClass, 'FontSize', 14, 'FontWeight', 'bold');
            case 3
                ylabel(ax, minorityClass, 'FontSize', 14, 'FontWeight', 'bold');
                xlabel(ax, majorityClass, 'FontSize', 14, 'FontWeight', 'bold');
            case 4
                xlabel(ax, minorityClass, 'FontSize', 14, 'FontWeight', 'bold');
            otherwise
                % keep default ticks/labels for interior tiles
        end
    end
end

% Create placeholders for the remaining tiles (7,8,15,16) so they have axes handles
for idx = 1:16
    if isempty(axesHandles(idx)) || ~isgraphics(axesHandles(idx))
        ax = nexttile(tBig, idx);
        % keep placeholder visually neutral for now (we'll populate them below)
        cla(ax);
        axesHandles(idx) = ax;
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

% --- ASSUME 2x2 confusion matrix (per your request). Error if not 2x2. ---
if ~(nRows == 2 && nCols == 2)
    error('confMatTest1 is not 2x2. This script assumes a 2x2 confusion matrix.');
end

% Define colors
blueColor = [0.2 0.6 0.8];
redColor  = [0.85 0.33 0.1];

% Label offset values (normalized units relative to each axis)
% these ensure labels sit further away from tick labels, matching the separation of other tiles
xLabelOffset = -0.18;   % negative => move down (for XLabel)
yLabelOffset = -0.21;   % negative => move left (for YLabel)

% Map 2x2 cells to individual tiles (top-left, top-right, bottom-left, bottom-right)
tileMap = [7, 8; 15, 16];   % top-left, top-right, bottom-left, bottom-right

% Loop cells and display counts with matching label formatting & colors
for r = 1:2
    for c = 1:2
        tileIndex = tileMap(r,c);
        axCell = axesHandles(tileIndex);

        cla(axCell);                      % clear placeholders but keep axes
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
            % left column -> show TRUE class as ylabel (black for contrast)
            ylabel(axCell, lblTrue, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');
            % move ylabel outward (normalized units)
            axCell.YLabel.Units = 'normalized';
            % center vertically and push left
            axCell.YLabel.Position = [yLabelOffset, 0.5, 0];
        end
        if r == 2
            % bottom row -> show PRED class as xlabel (black)
            xlabel(axCell, lblPred, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');
            % move xlabel outward (normalized units)
            axCell.XLabel.Units = 'normalized';
            axCell.XLabel.Position = [0.5, xLabelOffset, 0];
        end
    end
end

% --- Draw bounding boxes around each 2x2 group (so they read as 2x2 mini-grids) ---
%groupBoxes = {
%    [1, 2, 9, 10],   % Model 1
%    [3, 4, 11, 12],  % Model 2
%    [5, 6, 13, 14],  % Model 3
%    [7, 8, 15, 16]   % Test confusion matrix tiles
%};
%
% % Extra spacing parameters to make the rectangles wider and shifted left
%extraPadLeft  = 0.0085;  % how much extra to extend on the LEFT (normalized units)
%extraPadRight = 0.0;     % how much extra to extend on the RIGHT
%pad = 0.008;             % base padding used previously
%
%for g = 1:numel(groupBoxes)
%    tileIDs = groupBoxes{g};
%
%    % Collect positions for these tiles (use axes handles; they all exist)
%    posList = zeros(numel(tileIDs), 4);
%    validCount = 0;
%    for t = 1:numel(tileIDs)
%        idx = tileIDs(t);
%        if isgraphics(axesHandles(idx))
%            validCount = validCount + 1;
%            posList(validCount, :) = axesHandles(idx).Position;
%        end
%    end
%    if validCount == 0
%        continue;
%    end
%    posList = posList(1:validCount, :);
%
%    % Compute bounding rectangle in normalized figure coordinates
%    xMin = min(posList(:,1));
%    yMin = min(posList(:,2));
%    xMax = max(posList(:,1) + posList(:,3));
%    yMax = max(posList(:,2) + posList(:,4));
%    width = xMax - xMin;
%    height = yMax - yMin;
%
%    % Make rectangle wider and shift it left a bit. Clamp to [0,1].
%    rectLeft  = max(0, xMin - pad - extraPadLeft);
%    rectWidth = width + 2*pad + extraPadLeft + extraPadRight;
%    % ensure rectWidth doesn't overflow the right side of figure
%    if rectLeft + rectWidth > 1
%        rectWidth = 1 - rectLeft;
%    end
%    rectBottom = max(0, yMin - pad);
%    rectHeight = height + 2*pad;
%    if rectBottom + rectHeight > 1
%        rectHeight = 1 - rectBottom;
%    end
%    rectPos = [rectLeft, rectBottom, rectWidth, rectHeight];
%
%    % Draw rectangle annotation (keeps underlying axes intact and visible)
%    annotation(bigFig, 'rectangle', rectPos, ...
%        'Color', [0 0 0], 'LineWidth', 1.5, 'LineStyle', '-');
%
%    % Add a small centered title above the group (non-intrusive textbox)
%    switch g
%        case 1, labelStr = 'Model 1';
%        case 2, labelStr = 'Model 2';
%        case 3, labelStr = 'Model 3';
%        case 4, labelStr = 'Test Confusion';
%        otherwise, labelStr = '';
%    end
%    tbHeight = 0.035;
%    tbWidth  = rectPos(3) * 0.5;
%    tbX = rectPos(1) + (rectPos(3) - tbWidth)/2;
%    tbY = rectPos(2) + rectPos(4) + 0.002;  % place a little above the rectangle
%    annotation(bigFig, 'textbox', [tbX, tbY, tbWidth, tbHeight], ...
%        'String', labelStr, ...
%        'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
%        'EdgeColor','none', 'FontWeight','bold', 'FontSize', 11);
%end

% --- Global labels for the big layout ---
xlabel(tBig, 'Predicted Class', 'FontSize', 14, 'FontWeight', 'bold');
ylabel(tBig, 'Actual Class', 'FontSize', 14, 'FontWeight', 'bold');
title(tBig, 'Confusion Matrix Cell Distributions (All Models)', 'FontSize', 16);

% Save the figure:
savefig(bigFig, fullfile(savePath, char(json.outputFileNames.confusionMatrixGrouped)));



%% EXAMPLE CONFUSION CHART
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
