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


%% CROSS-VALIDATION TREE
% Once chosen we are going to use a Classification Tree model we have to
% make sure its predictive capacity is good enough

    % Divides the dataset into k equal-sized subdatasets and trains the 
    % model k times, each time leaving as the validation dataset one of 
    % said partitions. In this case it's going to perform the test 5 times,
    % each time using 80% as the training dataset and the other 20% as the
    % test dataset. It does not return a normal decision tree, but one used
    % to assess the quality of the model.

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
fprintf('5-fold CV loss: %.3f\n', missClassRate);


%% CROSS-VALIDATION TREE WITH WEIGHTS
    % Because the dataset's classes are very imbalanced the model will be
    % biased thowards the majority class, making it more likely to make a
    % mistake in the minority class (our interest class). To manage imbalance
    % we are going to use the Weights Approach.


% ------------------ 1) PREPARE WEIGHTS ------------------ 
% --- Compute class frequencies ---
[classNames, ~, idxClass] = unique(T_ResultsVariable);      % classNames is a cell array of labels
nClasses = numel(classNames);                               % number of classes
counts   = accumarray(idxClass, 1);                         % count per class
nTotal   = numel(T_ResultsVariable);                        % total number of observations

% --- Compute per-class weights: w_c ∝ 1/counts(c) ---
% Normalize so that sum of all sample weights = nTotal
invFreq   = 1 ./ counts;                                    % inverse frequency per class
normFactor = nTotal / sum(counts .* invFreq);
classWeights = normFactor .* invFreq;                       % vector of length nClasses

% --- Build a weight vector W matching observations ---
W = classWeights(idxClass);                 % maps each sample to its class weight

% --- (Optional) Check that sum(W) == nTotal: ---
%fprintf('Sum of weights: %.1f (should be %d)\n', sum(W), nTotal);


% ------------------ 2) WEIGHTED TREE ------------------ 
WeightCVMdl = fitctree( ...
    T_Data, T_ResultsVariable, ...
    'KFold',           5, ...
    'Weights',         W, ...
    'CategoricalPredictors', {'Genotype'}, ...
    'MinParentSize',   3);


% ------------------ 3) MODEL'S RESULTS ------------------------
wtLabel  = kfoldPredict(WeightCVMdl);

% --- Misclassification rate ---
missClassRateWeight  = kfoldLoss(WeightCVMdl); 
fprintf('Weighted   5-fold CV loss: %.3f\n', missClassRateWeight);



%% CROSS-VALIDATION TREE WITH OVERSAMPLING
    % Weighting didn´t work very well. I'm going to try oversampling as to
    % manage the imbalance and reduce error in the minority class.

% --- Create 5-fold partition ---
cv = cvpartition(T_ResultsVariable, 'KFold', 5);

% --- Initialize prediction vector ---
OSLabels = strings(size(T_ResultsVariable));

for i = 1:cv.NumTestSets
    % Indices for training and testing
    trainIdx = training(cv, i);
    testIdx  = test(cv, i);

    % Split data
    XTrain = T_Data(trainIdx,:);
    YTrain = T_ResultsVariable(trainIdx);
    XTest = T_Data(testIdx,:);
    YTest = T_ResultsVariable(testIdx);

    % ------------------ OVERSAMPLING -------------------
    % Find class counts
    classNames = unique(YTrain);
    counts = groupcounts(YTrain);
    maxCount = max(counts);

    XBalanced = XTrain;
    YBalanced = YTrain;

    for j = 1:numel(classNames)
        cls = classNames(j);
        idx = find(strcmp(YTrain, cls));

        % Compute how many more samples needed
        nToAdd = maxCount - numel(idx);

        if nToAdd > 0
            % Randomly sample with replacement
            sampledIdx = datasample(idx, nToAdd, 'Replace', true);
            XBalanced = [XBalanced; XTrain(sampledIdx,:)];
            YBalanced = [YBalanced; YTrain(sampledIdx)];
        end
    end

    % ------------------ Train model --------------------
    OSCVMdl = fitctree(...
        XBalanced, YBalanced, ...
        'CategoricalPredictors', {'Genotype'}, ...
        'MinParentSize', 3);

    % ------------------ Predict ------------------------
    YPred = predict(OSCVMdl, XTest);

    % Save predictions
    OSLabels(testIdx) = YPred;
end

% ------------------ MODEL'S RESULTS ------------------------
OSLabels = cellstr(OSLabels);

% --- Misclassification rate ---
missClassRateOS = sum(~strcmp(OSLabels, T_ResultsVariable)) / numel(T_ResultsVariable);
fprintf('Oversampled CV Misclassification Rate: %.3f\n', missClassRateOS);


%% TRAIN THE MODEL
% Using the whole of the dataset train the model in order to have a
% Classification Tree with predictive capacity.

Mdl = fitctree(T_Data, T_ResultsVariable, 'CategoricalPredictors', {'Genotype'}, 'MinParentSize',3);

% VIEW THE TREE
view(Mdl,'Mode','graph');
savefig(fullfile(savePath, 'DecisionTree.fig'));

% PREDICTORS' IMPORTANCE
    % Get importance and predictor names
    imp = predictorImportance(Mdl);
    predictorNames = Mdl.PredictorNames;
    
    % --- If you want to Sort by importance (descending) --- make this two lines readable
    %[impSorted, idxSorted] = sort(imp, 'descend');
    %predictorNamesSorted = predictorNames(idxSorted);
    
    % Plot sorted importance
    figure;
    bar(imp);
    %bar(impSorted); --- to sort by importance
    title('Predictor Importance Estimates');
    ylabel('Importance');
    xlabel('Predictors');
    
    % Set X-axis labels
    ax = gca;
    ax.XTick = 1:numel(predictorNames);
    %ax.XTick = 1:numel(predictorNamesSorted); --- to sort by importance
    ax.XTickLabel = predictorNames;
    %ax.XTickLabel = predictorNamesSorted; --- to sort by importance
    ax.XTickLabelRotation = 45;
    ax.TickLabelInterpreter = 'none';

    % save as .fig
    savefig(fullfile(savePath, 'PredictorImportance.fig'));

%% TEST DATASET
DeathFUS = predict(Mdl,T_Data);

% ------------------ Evaluate model ------------------------
% Compare predictions with true labels
nTotal = numel(T_ResultsVariable);
nIncorrect = sum(~strcmp(DeathFUS, T_ResultsVariable));  % Count incorrect predictions

% Compute test error (misclassification rate)
testError = nIncorrect / nTotal;

% Display result
fprintf('Test Error (on training data): %.3f\n', testError);


%% PREDICT WITH ANOTHER DATASET
% I'll have to write this section when I have another dataset to try it
% with


%% CONFUSION CHART
% Ensure consistent types
trueLabels = string(T_ResultsVariable);
Label = string(Label);
wtLabel = string(wtLabel);
OSLabels = string(OSLabels);
trainingDatasetPredictions = string(DeathFUS);

% Create a new figure
figure('Units', 'normalized', 'Position', [0.1 0.3 0.8 0.4]);
set(gca,'DataAspectRatio',[6 .5 0.2]);
sgtitle ('Confussion Matrices per model');

    % --- 1. Standard Confusion Matrix ---
    subplot(1,4,1);
    confusionchart(trueLabels, Label);
    title('Standard 5-fold cross-validation');
      
    % --- 2. Weighted Confusion Matrix ---
    subplot(1,4,2);
    confusionchart(trueLabels, wtLabel);
    title('Weighted 5-f cv');
    
    % --- 3. Oversampled Confusion Matrix ---
    subplot(1,4,3);
    confusionchart(trueLabels, OSLabels);
    title('Oversampled 5-f cv');
    
    % --- 4. Final Model on Training Set ---
    subplot(1,4,4);
    confusionchart(trueLabels, trainingDatasetPredictions);
    title('Final Model (with Training Data)');

 % save as .fig
 savefig(fullfile(savePath, 'ConfusionCharts.fig'));


 %% CREATE AN EXCEL FILE 