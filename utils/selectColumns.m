function T_Data = selectColumns (T_Original, Target_columns, Ignore_columns)

% SELECTCOLUMNS Selects the columns in which you want to perform the
% analysis. You can either enter the common string in all column names or
% the index ranges of the selected columns
% 
% INPUTS:
%   T_Original       - Table (nSamples x nRegions), each column is a region.
%   Target_columns   - String or numerical array in which odd positions are beginings and even positions are endings of the desired intevals.
%   Ingnore_columns  - Numerical array with column indexes to be ingnored.
%
% OUTPUT:
%   T_Data           - Filtered table.



    if ischar (Target_columns) || isstring (Target_columns)
        T_Data = T_Original(:,contains(T_Original.Properties.VariableNames,Target_columns));
    elseif isvector (Target_columns) & (mod(Target_columns, 1) == 0) & isvector (Ignore_columns) & (mod(Ignore_columns, 1) == 0)
        assert (mod(length (Target_columns),2) == 0, 'Error in Target_columns: indices must be provided in pairs')
        aux = [];
        for i = 1:2:length(Target_columns)
            aux = [aux, Target_columns(i): Target_columns(i+1)];
        end
        aux = unique(aux);
        Target_columns = setdiff(aux,Ignore_columns);
        T_Data = T_Original(:,Target_columns);
    else
        error('selecctColumns format check failed')
    end
end