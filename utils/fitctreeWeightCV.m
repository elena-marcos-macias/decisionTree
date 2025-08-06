function [WeightCVMdl, W, classWeights, classNames] = fitctreeWeightCV(T_Data, T_ResultsVariable, numKFold, catPredictors, minParentSize)
% fitWeightedTreeCV - Fits a weighted decision tree using inverse class frequencies
%
% Inputs:
%   T_Data            - Table of predictor variables
%   T_ResultsVariable - Vector of class labels (categorical or string/cell array)
%   catPredictors     - Cell array of names of categorical predictors (e.g., {'Genotype'})
%   minParentSize     - Minimum number of observations per branch split (e.g., 3)
%
% Outputs:
%   WeightCVMdl   - Trained cross-validated weighted decision tree (ClassificationPartitionedModel)
%   W             - Weights assigned to each observation
%   classWeights  - Weights per class (for inspection)
%   classNames    - Names of classes (cell array)

    % --- Compute class frequencies ---
    [classNames, ~, idxClass] = unique(T_ResultsVariable);  % classNames is a cell array
    nClasses = numel(classNames);
    counts   = accumarray(idxClass, 1);
    nTotal   = numel(T_ResultsVariable);

    % --- Inverse frequency weights ---
    invFreq    = 1 ./ counts;
    normFactor = nTotal / sum(counts .* invFreq);
    classWeights = normFactor .* invFreq;

    % --- Assign weights to samples ---
    W = classWeights(idxClass);

    % --- Train weighted cross-validated tree ---
    WeightCVMdl = fitctree( ...
        T_Data, T_ResultsVariable, ...
        'KFold',           numKFold, ...
        'Weights',         W, ...
        'CategoricalPredictors', catPredictors, ...
        'MinParentSize',   minParentSize);

end