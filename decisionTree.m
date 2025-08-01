%--------- Add function path and set save path ----------
addpath (genpath('./utils'));
addpath ('./requirements');
savePath = './results/';

%--------- Load data ---------
fileName = char("FUS_test6.xlsx");
T_Original = readtable(['./data/' fileName]);

% -------- Remove rows with NaN --------
T_Original = rmmissing(T_Original);  % Remove rows that contain at least one NaN

% -------- Select data to use (columns with string or numerical criteria) ---------------
target_columns = [1,1,4,22];
ignore_columns = [2,3];
T_Data = selectColumns (T_Original, target_columns, ignore_columns);


% ---------- Apply decision Tree model -----------
Mdl = fitctree(T_Data,T_Data.Death);