@echo off
setlocal

rem ============================================================================
rem Tune Random Forest and Logistic Regression on Windows with 16 GB RAM.
rem
rem Fixed method settings:
rem   features      = concat
rem   node features = 18
rem   dimensions    = 100, 500, 1000
rem
rem Each classifier search includes class weighting, so balancing is selected
rem by Optuna. Experiments run one at a time with one CPU thread.
rem ============================================================================

cd /d "%~dp0"

set "D=msc_new/AID_1159555_datatable.csv_pivot_class_std_smi_no_stereochem_no_salt.data"
set "RESULTS=results_windows_other_classifiers"
set "TRIALS=50"

set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "PYTHONUNBUFFERED=1"

echo ============================================================================
echo Checking Python, packages, tuning script and input files
echo ============================================================================

python --version
if errorlevel 1 goto :failed

rem Refuse the Python 3.9 environment used by the earlier Windows method runs.
rem Classifier comparisons are meaningful only when they share one environment.
python -c "import sys, numpy, pandas, torch, torch_geometric, optuna, rdkit, sklearn; assert sys.version_info[:2] == (3, 8), 'activate the Python 3.8 environment'; assert numpy.__version__ == '1.24.4'; assert pandas.__version__ == '1.4.4'; assert sklearn.__version__ == '1.0.2'; assert torch.__version__.startswith('2.0.1'); print('Python/package versions: OK'); print('CUDA available:', torch.cuda.is_available())"
if errorlevel 1 goto :failed

if not exist "Run_other_classifier_tuning.py" (
    echo ERROR: Run_other_classifier_tuning.py was not found.
    goto :failed
)

if not exist "%D%.tr_0.2_1.txt" (
    echo ERROR: training file not found: %D%.tr_0.2_1.txt
    goto :failed
)

if not exist "%D%.te_0.2_1.txt" (
    echo ERROR: test file not found: %D%.te_0.2_1.txt
    goto :failed
)

rem Random Forest searches.
call :run "01_rf_concat_nf18_dim100"  "rf" 100
if errorlevel 1 goto :failed

call :run "02_rf_concat_nf18_dim500"  "rf" 500
if errorlevel 1 goto :failed

call :run "03_rf_concat_nf18_dim1000" "rf" 1000
if errorlevel 1 goto :failed

rem Logistic Regression searches.
call :run "04_lr_concat_nf18_dim100"  "lr" 100
if errorlevel 1 goto :failed

call :run "05_lr_concat_nf18_dim500"  "lr" 500
if errorlevel 1 goto :failed

call :run "06_lr_concat_nf18_dim1000" "lr" 1000
if errorlevel 1 goto :failed

echo.
echo ============================================================================
echo ALL RF/LR TUNING EXPERIMENTS COMPLETED
echo Results: %RESULTS%
echo Best parameters: each result folder under configs\ED
echo ============================================================================
exit /b 0


rem ============================================================================
rem Run one RF/LR Optuna search.
rem Arguments: result folder name, classifier, dimension
rem ============================================================================
:run
set "RUN_NAME=%~1"
set "CLF=%~2"
set "DIM=%~3"
set "OUT=%RESULTS%\%RUN_NAME%"

echo.
echo ============================================================================
echo Running %RUN_NAME%
echo classifier=%CLF%  features=concat  node_features=18  dim=%DIM%
echo trials=%TRIALS%  output=%OUT%
echo ============================================================================

rem Skip configurations that completed during an earlier invocation.
if exist "%OUT%\all_results.csv" (
    echo SKIP: this configuration is already complete.
    exit /b 0
)

python Run_other_classifier_tuning.py ^
  -i "%D%.tr_0.2_1.txt" ^
  -p "%D%.te_0.2_1.txt" ^
  --group ED ^
  --features concat ^
  --node-features 18 ^
  --dim %DIM% ^
  --clf %CLF% ^
  --trials %TRIALS% ^
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
echo ERROR: the RF/LR tuning batch stopped because the previous command failed.
echo Completed result folders and shared caches were preserved.
echo Fix the error and run this file again; completed configurations are skipped.
echo ============================================================================
exit /b 1
