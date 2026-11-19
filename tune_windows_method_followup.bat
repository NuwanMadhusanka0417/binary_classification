@echo off
setlocal

rem ============================================================================
rem Safe follow-up method sweep for a Windows PC with 16 GB RAM.
rem
rem Run AFTER tune_windows_commands.bat:
rem     tune_windows_method_followup.bat
rem
rem This file uses one representative ED dataset, one CPU thread, and runs one
rem experiment at a time. It does NOT run HTSFP10 or dimensions 5000/10000.
rem Every configuration is stored in its own results folder.
rem ============================================================================

cd /d "%~dp0"

set "D=msc_new/AID_1159555_datatable.csv_pivot_class_std_smi_no_stereochem_no_salt.data"
set "RESULTS=results_windows_method"

rem Leave this empty for unbalanced runs.
rem If the earlier classifier test showed that balancing is better, replace it
rem with: set "BALANCE_ARG=--balance"
set "BALANCE_ARG="

rem Conservative CPU/RAM settings for a 16 GB computer.
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "PYTHONUNBUFFERED=1"

echo ============================================================================
echo Checking Python, packages and input files
echo ============================================================================

python --version
if errorlevel 1 goto :failed

python -c "import torch, torch_geometric, xgboost, optuna, rdkit, sklearn; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('required imports: OK')"
if errorlevel 1 goto :failed

if not exist "%D%.tr_0.2_1.txt" (
    echo ERROR: training file not found: %D%.tr_0.2_1.txt
    goto :failed
)

if not exist "%D%.te_0.2_1.txt" (
    echo ERROR: test file not found: %D%.te_0.2_1.txt
    goto :failed
)

rem ============================================================================
rem Dimension 100: compare GVFA, concat and bundle with 7 and 18 node features.
rem ============================================================================

call :run "01_gvfa_nf7_dim100"   "gvfa"   7 100
if errorlevel 1 goto :failed

call :run "02_concat_nf7_dim100" "concat" 7 100
if errorlevel 1 goto :failed

call :run "03_bundle_nf7_dim100" "bundle" 7 100
if errorlevel 1 goto :failed

call :run "04_gvfa_nf18_dim100"   "gvfa"   18 100
if errorlevel 1 goto :failed

call :run "05_concat_nf18_dim100" "concat" 18 100
if errorlevel 1 goto :failed

call :run "06_bundle_nf18_dim100" "bundle" 18 100
if errorlevel 1 goto :failed

rem ============================================================================
rem Dimension 500: repeat the same method comparison.
rem ============================================================================

call :run "07_gvfa_nf7_dim500"   "gvfa"   7 500
if errorlevel 1 goto :failed

call :run "08_concat_nf7_dim500" "concat" 7 500
if errorlevel 1 goto :failed

call :run "09_bundle_nf7_dim500" "bundle" 7 500
if errorlevel 1 goto :failed

call :run "10_gvfa_nf18_dim500"   "gvfa"   18 500
if errorlevel 1 goto :failed

call :run "11_concat_nf18_dim500" "concat" 18 500
if errorlevel 1 goto :failed

call :run "12_bundle_nf18_dim500" "bundle" 18 500
if errorlevel 1 goto :failed

rem ============================================================================
rem Dimension 1000: one confirmation run for the expected strongest method.
rem This starts only after every dimension-100/500 run succeeds.
rem ============================================================================

call :run "13_concat_nf18_dim1000" "concat" 18 1000
if errorlevel 1 goto :failed

echo.
echo ============================================================================
echo ALL SAFE FOLLOW-UP EXPERIMENTS COMPLETED
echo Results: %RESULTS%
echo Best Optuna parameters: each folder under configs\ED
echo ============================================================================
exit /b 0


rem ============================================================================
rem Run one tuned XGBoost configuration.
rem Arguments: result name, feature type, node features, dimension
rem ============================================================================
:run
set "RUN_NAME=%~1"
set "FEATURE=%~2"
set "NODE_FEATURES=%~3"
set "DIM=%~4"
set "OUT=%RESULTS%\%RUN_NAME%"

echo.
echo ============================================================================
echo Running %RUN_NAME%
echo features=%FEATURE%  node_features=%NODE_FEATURES%  dim=%DIM%
echo output=%OUT%
echo ============================================================================

rem A completed run has an all_results.csv file. Skip it when restarting.
if exist "%OUT%\all_results.csv" (
    echo SKIP: this configuration is already complete.
    exit /b 0
)

python Run_experiment.py ^
  -i "%D%.tr_0.2_1.txt" ^
  -p "%D%.te_0.2_1.txt" ^
  --group ED ^
  --features %FEATURE% ^
  --node-features %NODE_FEATURES% ^
  --dim %DIM% ^
  --clf xgb ^
  %BALANCE_ARG% ^
  --tune 50 ^
  --folds 10 ^
  --seed 42 ^
  --n-jobs 1 ^
  --results-dir "%OUT%"
if errorlevel 1 exit /b 1

python collect_results.py --results-dir "%OUT%" --latest
if errorlevel 1 exit /b 1

exit /b 0


:failed
echo.
echo ============================================================================
echo ERROR: the follow-up batch stopped because the previous command failed.
echo Completed result folders and shared cache files have been preserved.
echo Fix the error and run this batch again; completed configurations are skipped.
echo ============================================================================
exit /b 1
