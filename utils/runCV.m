function [errors, cm, fp, tp, auc] = runCV(modelType, T_Data, T_ResultsVariable, ...
                                           nFolds, catVariable, majorityClass, minorityClass, trueBinary)
    switch modelType
        case 'Standard'
            CVMdl = fitctree(T_Data, T_ResultsVariable, ...
                'KFold', nFolds, ...
                'CategoricalPredictors', {catVariable}, ...
                'MinParentSize', 3);
            [Label, Score] = kfoldPredict(CVMdl);

        case 'Weighted'
            CVMdl = fitctreeWeightCV(T_Data, T_ResultsVariable, nFolds, {catVariable}, 3);
            [Label, Score] = kfoldPredict(CVMdl);

        case 'Oversampled'
            [Label, Score] = kfoldPredictOS(T_Data, T_ResultsVariable, nFolds, {catVariable}, 3);
    end

    missClassRate = mean(~strcmp(Label, T_ResultsVariable));
    [missMajority, missMinority] = classwiseMisclassification(T_ResultsVariable, Label, majorityClass, minorityClass);
    [fp,tp,~,auc] = perfcurve(trueBinary, Score(:,2), 1);

    errors = [missClassRate, missMajority, missMinority, auc];
    cm = confusionmat(T_ResultsVariable, string(Label));
end