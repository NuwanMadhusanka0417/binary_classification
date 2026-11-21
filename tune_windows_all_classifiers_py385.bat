@echo off
setlocal

rem ============================================================================
rem Tune RF, LR and XGBoost with Python 3.8.5 on a 16 GB Windows computer.
rem
rem Option A: activate your Python 3.8.5 environment before running this file.
rem Option B: set its path first:
rem     set "PY385_ENV=C:\path\to\your\venv"
rem     tune_windows_all_classifiers_py385.bat
rem
rem Fixed representation: concat, 18 node features, dimension 1000.
rem Runs are sequential, use one CPU thread, and save to separate folders.
rem ============================================================================

cd /d "%~dp0"

if defined PY385_ENV (
    if not exist "%PY385_ENV%\Scripts\activate.bat" (
        echo ERROR: cannot find %PY385_ENV%\Scripts\activate.bat
        exit /b 1
    )
    call "%PY385_ENV%\Scripts\activate.bat"
)

set "D=msc_new/AID_1159555_datatable.csv_pivot_class_std_smi_no_stereochem_no_salt.data"
set "RESULTS=results_tuned_dim1000_windows"
set "TRIALS=50"

set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "PYTHONUNBUFFERED=1"

echo ============================================================================
echo Checking the Python 3.8.5 environment
echo ============================================================================

python -c "import sys; print(sys.version); assert sys.version_info[:3] == (3, 8, 5), 'Python 3.8.5 is required'"
if errorlevel 1 goto :failed

python -c "import numpy, pandas, scipy, sklearn, torch, torch_geometric, xgboost, optuna, rdkit; print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('sklearn', sklearn.__version__); print('torch', torch.__version__); print('xgboost', xgboost.__version__); print('CUDA available:', torch.cuda.is_available()); print('required imports: OK')"
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

call :run "01_rf_concat_nf18_dim1000" "rf"
if errorlevel 1 goto :failed

call :run "02_lr_concat_nf18_dim1000" "lr"
if errorlevel 1 goto :failed

call :run "03_xgb_concat_nf18_dim1000" "xgb"
if errorlevel 1 goto :failed

echo.
echo ============================================================================
echo ALL CLASSIFIER TUNING RUNS COMPLETED
echo Results: %RESULTS%
echo Best parameters: each result folder under configs\ED
echo ============================================================================
exit /b 0


:run
set "RUN_NAME=%~1"
set "CLF=%~2"
set "OUT=%RESULTS%\%RUN_NAME%"

echo.
echo ============================================================================
echo Running %RUN_NAME%
echo classifier=%CLF%  features=concat  node_features=18  dim=1000
echo trials=%TRIALS%  output=%OUT%
echo ============================================================================

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
  --dim 1000 ^
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
echo ERROR: the batch stopped because the command above failed.
echo Activate the Python 3.8.5 environment, fix the error, and run it again.
echo Completed result folders and shared caches are preserved and skipped.
echo ============================================================================
exit /b 1
