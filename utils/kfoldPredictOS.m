function [OSLabels, OSScores] = kfoldPredictOS (T_Data, T_ResultsVariable, numKFold, catPredictors, minParentSize)
% fitctreeOSCV - Performs K-fold cross-validation with random oversampling for class balance
%
% Inputs:
%   T_Data            - Table of predictor variables
%   T_ResultsVariable - Vector of class labels (categorical, string, or cell array)
%   numKFold          - Number of folds for cross-validation
%   catPredictors     - Cell array of categorical predictor names (e.g., {'Genotype'})
%   minParentSize     - Minimum parent size for decision tree splits (e.g., 3)
%
% Outputs:
%   OSLabels          - Cell array of predicted labels (same size as T_ResultsVariable)
%   OSScores          - Nx2 matrix of predicted class scores (posterior probabilities)

    % Create K-fold partition
    cv = cvpartition(T_ResultsVariable, 'KFold', numKFold);

    % Initialize outputs
    OSLabels = strings(size(T_ResultsVariable));
    OSScores = zeros(numel(T_ResultsVariable), 2);  % Assuming binary classification

    for i = 1:cv.NumTestSets
        % Indices for this fold
        trainIdx = training(cv, i);
        testIdx  = test(cv, i);

        % Extract training and test sets
        XTrain = T_Data(trainIdx, :);
        YTrain = T_ResultsVariable(trainIdx);
        XTest  = T_Data(testIdx, :);

        % Oversample minority class
        classNames = unique(YTrain);
        counts = groupcounts(YTrain);
        maxCount = max(counts);

        XBalanced = XTrain;
        YBalanced = YTrain;

        for j = 1:numel(classNames)
            cls = classNames(j);
            idx = find(strcmp(YTrain, cls));

            nToAdd = maxCount - numel(idx);

            if nToAdd > 0
                sampledIdx = datasample(idx, nToAdd, 'Replace', true);
                XBalanced = [XBalanced; XTrain(sampledIdx, :)];
                YBalanced = [YBalanced; YTrain(sampledIdx)];
            end
        end

        % Train decision tree
        OSCVMdl = fitctree( ...
            XBalanced, YBalanced, ...
            'CategoricalPredictors', catPredictors, ...
            'MinParentSize', minParentSize);

        % Predict labels and scores
        [YPred, score] = predict(OSCVMdl, XTest);

        % Store predictions
        OSLabels(testIdx)  = YPred;
        OSScores(testIdx,:) = score;
    end

    % Format labels
    OSLabels = cellstr(OSLabels);
    
end