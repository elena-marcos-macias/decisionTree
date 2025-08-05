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


%% VALIDATE THE MODEL:
% Once chosen we are going to use a Classification Tree model we have to
% make sure its predictive capacity is good enough

% TRAIN THE CROSS-VALIDATION MODEL
    % Divides the dataset into k equal-sized subdatasets and trains the 
    % model k times, each time leaving as the validation dataset one of 
    % said partitions. In this case it's going to perform the test 5 times,
    % each time using 80% as the training dataset and the other 20% as the
    % test dataset. It does not return a normal decision tree, but one used
    % to assess the quality of the model.

CVMdl = fitctree(T_Data,T_ResultsVariable, 'KFold', 5, 'CategoricalPredictors', {'Genotype'}, 'MinParentSize',3); 

% ASSESS THE QUALITY OF THE MODEL

[label, Score] = kfoldPredict(CVMdl);
    CVLoss = kfoldLoss(CVMdl);
    confusionchart (T_ResultsVariable,label);

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
