%% Helper function to calculate class-wise misclassification rates
function [missMajority, missMinority] = classwiseMisclassification(trueLabels, predictedLabels, majorityClass, minorityClass)
    trueLabels = string(trueLabels);
    predictedLabels = string(predictedLabels);

    majorityMask = (trueLabels == majorityClass);
    majorityTotal = sum(majorityMask);
    majorityWrong = sum(predictedLabels(majorityMask) ~= majorityClass);
    missMajority = majorityWrong / majorityTotal;

    minorityMask = (trueLabels == minorityClass);
    minorityTotal = sum(minorityMask);
    minorityWrong = sum(predictedLabels(minorityMask) ~= minorityClass);
    missMinority = minorityWrong / minorityTotal;
end