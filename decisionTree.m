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

%% PREPARE WEIGHTS
% Because the dataset's classes are very imbalanced the model will be
% biased thowards the majority class, making it more likely to make a
% mistake in the minority class (our interest class). To manage imbalance
% we are going to use the Weights Approach.

% --- 1. Compute class frequencies ---
[classNames, ~, idxClass] = unique(T_ResultsVariable);      % classNames is a cell array of labels
nClasses = numel(classNames);                               % number of classes
counts   = accumarray(idxClass, 1);                         % count per class
nTotal   = numel(T_ResultsVariable);                        % total number of observations

% --- 2. Compute per-class weights: w_c ∝ 1/counts(c) ---
%    Normalize so that sum of all sample weights = nTotal
invFreq   = 1 ./ counts;                                    % inverse frequency per class
normFactor = nTotal / sum(counts .* invFreq);
classWeights = normFactor .* invFreq;                       % vector of length nClasses

% --- 3. Build a weight vector W matching observations ---
W = classWeights(idxClass);                 % maps each sample to its class weight

% (Optional) Check that sum(W) == nTotal:
fprintf('Sum of weights: %.1f (should be %d)\n', sum(W), nTotal);


%% VALIDATE THE MODEL
% Once chosen we are going to use a Classification Tree model we have to
% make sure its predictive capacity is good enough

% TRAIN THE CROSS-VALIDATION MODEL
    % Divides the dataset into k equal-sized subdatasets and trains the 
    % model k times, each time leaving as the validation dataset one of 
    % said partitions. In this case it's going to perform the test 5 times,
    % each time using 80% as the training dataset and the other 20% as the
    % test dataset. It does not return a normal decision tree, but one used
    % to assess the quality of the model.

   
% 1) Unweighted CV tree
CVMdl = fitctree( ...
    T_Data, T_ResultsVariable, ...
    'KFold',           5, ...
    'CategoricalPredictors', {'Genotype'}, ...
    'MinParentSize',   3);

% 2) Weighted CV tree
%    — note the correct name: 'Weights'
WeightCVMdl = fitctree( ...
    T_Data, T_ResultsVariable, ...
    'KFold',           5, ...
    'Weights',         W, ...
    'CategoricalPredictors', {'Genotype'}, ...
    'MinParentSize',   3);

% 3) Compute losses 
    % --- missclassification rates ----
unweightedCVLoss = kfoldLoss(CVMdl);
weightedCVLoss   = kfoldLoss(WeightCVMdl);

fprintf('Unweighted 5-fold CV loss: %.3f\n', unweightedCVLoss);
fprintf('Weighted   5-fold CV loss: %.3f\n', weightedCVLoss);

    % --- Hinge Loss Calculation 


    
% 4) (Optional) confusion charts
unwLabel = kfoldPredict(CVMdl);
wtLabel  = kfoldPredict(WeightCVMdl);

figure;
confusionchart(T_ResultsVariable, unwLabel);
title('Unweighted CV Confusion Matrix');

figure;
confusionchart(T_ResultsVariable, wtLabel);
title('Weighted CV Confusion Matrix');


%% TRAIN THE MODEL
% Using the whole of the dataset train the model in order to have a
% Classification Tree with predictive capacity.

Mdl = fitctree(T_Data, T_ResultsVariable, 'CategoricalPredictors', {'Genotype'}, 'MinParentSize',3);

% VIEW THE TREE
view(Mdl,'Mode','graph');

% PREDICTORS' IMPORTANCE
imp = predictorImportance(Mdl);
    
    % En esta figura tiene que haber un error porque no posiciona las
    % barras en los predictores que ha usado para crear los nodos.
    figure;
    bar(imp);
    title('Predictor Importance Estimates');
    ylabel('Estimates');
    xlabel('Predictors');
    h = gca;
    h.XTickLabel = Mdl.PredictorNames;
    h.XTickLabelRotation = 45;
    h.TickLabelInterpreter = 'none';

%% TEST DATASET
DeathFUS = predict(Mdl,T_Data);

%CALCULATE TEST ERROR

% Compare predictions with true labels
nTotal = numel(T_ResultsVariable);
nIncorrect = sum(~strcmp(DeathFUS, T_ResultsVariable));  % Count incorrect predictions

% Compute test error (misclassification rate)
testError = nIncorrect / nTotal;

% Display result
fprintf('Test Error (on training data): %.3f\n', testError);
