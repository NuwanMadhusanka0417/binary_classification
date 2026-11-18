@echo off
setlocal

rem ============================================================================
rem Windows Command Prompt batch file for a 16 GB CPU-only computer.
rem
rem Run from the project directory with:
rem     tune_windows_commands.bat
rem
rem Experiments run ONE AT A TIME. The batch stops immediately if one fails.
rem XGBoost is the only classifier fine-tuned by the current Python code.
rem Random Forest and Logistic Regression are fixed comparison models.
rem ============================================================================

cd /d "%~dp0"

rem Dataset base path: do not include the .tr_0.2_1.txt or .te_0.2_1.txt suffix.
set "D=msc_new/AID_1159555_datatable.csv_pivot_class_std_smi_no_stereochem_no_salt.data"

rem Conservative settings for 16 GB RAM.
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "PYTHONUNBUFFERED=1"

set "RESULTS=results_windows_cmd"

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
rem STEP 1: Untuned XGBoost baseline
rem Saved in results_windows_cmd\01_xgb_rdkit_untuned
rem ============================================================================

echo.
echo [STEP 1/9] Untuned XGBoost baseline
python Run_experiment.py ^
  -i "%D%.tr_0.2_1.txt" ^
  -p "%D%.te_0.2_1.txt" ^
  --group ED ^
  --features rdkit ^
  --clf xgb ^
  --tune 0 ^
  --folds 10 ^
  --seed 42 ^
  --n-jobs 1 ^
  --results-dir "%RESULTS%\01_xgb_rdkit_untuned"
if errorlevel 1 goto :failed

python collect_results.py --results-dir "%RESULTS%\01_xgb_rdkit_untuned" --latest
if errorlevel 1 goto :failed


rem ============================================================================
rem STEP 2: Fine-tuned XGBoost without class balancing
rem Optuna jointly searches all eight XGBoost parameters for 50 trials.
rem Saved in results_windows_cmd\02_xgb_rdkit_tuned
rem ============================================================================

echo.
echo [STEP 2/9] Fine-tuned XGBoost without class balancing
python Run_experiment.py ^
  -i "%D%.tr_0.2_1.txt" ^
  -p "%D%.te_0.2_1.txt" ^
  --group ED ^
  --features rdkit ^
  --clf xgb ^
  --tune 50 ^
  --folds 10 ^
  --seed 42 ^
  --n-jobs 1 ^
  --results-dir "%RESULTS%\02_xgb_rdkit_tuned"
if errorlevel 1 goto :failed

python collect_results.py --results-dir "%RESULTS%\02_xgb_rdkit_tuned" --latest
if errorlevel 1 goto :failed


rem ============================================================================
rem STEP 3: Fine-tuned XGBoost with class balancing
rem Saved in results_windows_cmd\03_xgb_rdkit_tuned_balanced
rem ============================================================================

echo.
echo [STEP 3/9] Fine-tuned XGBoost with class balancing
python Run_experiment.py ^
  -i "%D%.tr_0.2_1.txt" ^
  -p "%D%.te_0.2_1.txt" ^
  --group ED ^
  --features rdkit ^
  --clf xgb ^
  --balance ^
  --tune 50 ^
  --folds 10 ^
  --seed 42 ^
  --n-jobs 1 ^
  --results-dir "%RESULTS%\03_xgb_rdkit_tuned_balanced"
if errorlevel 1 goto :failed

python collect_results.py --results-dir "%RESULTS%\03_xgb_rdkit_tuned_balanced" --latest
if errorlevel 1 goto :failed


rem ============================================================================
rem STEP 4: Random Forest fixed baseline, without balancing
rem Saved in results_windows_cmd\04_rf_rdkit
rem ============================================================================

echo.
echo [STEP 4/9] Random Forest baseline
python Run_experiment.py ^
  -i "%D%.tr_0.2_1.txt" ^
  -p "%D%.te_0.2_1.txt" ^
  --group ED ^
  --features rdkit ^
  --clf rf ^
  --folds 10 ^
  --seed 42 ^
  --n-jobs 1 ^
  --results-dir "%RESULTS%\04_rf_rdkit"
if errorlevel 1 goto :failed

python collect_results.py --results-dir "%RESULTS%\04_rf_rdkit" --latest
if errorlevel 1 goto :failed


rem ============================================================================
rem STEP 5: Random Forest fixed baseline, with balancing
rem Saved in results_windows_cmd\05_rf_rdkit_balanced
rem ============================================================================

echo.
echo [STEP 5/9] Balanced Random Forest baseline
python Run_experiment.py ^
  -i "%D%.tr_0.2_1.txt" ^
  -p "%D%.te_0.2_1.txt" ^
  --group ED ^
  --features rdkit ^
  --clf rf ^
  --balance ^
  --folds 10 ^
  --seed 42 ^
  --n-jobs 1 ^
  --results-dir "%RESULTS%\05_rf_rdkit_balanced"
if errorlevel 1 goto :failed

python collect_results.py --results-dir "%RESULTS%\05_rf_rdkit_balanced" --latest
if errorlevel 1 goto :failed


rem ============================================================================
rem STEP 6: Logistic Regression fixed baseline, without balancing
rem Saved in results_windows_cmd\06_lr_rdkit
rem ============================================================================

echo.
echo [STEP 6/9] Logistic Regression baseline
python Run_experiment.py ^
  -i "%D%.tr_0.2_1.txt" ^
  -p "%D%.te_0.2_1.txt" ^
  --group ED ^
  --features rdkit ^
  --clf lr ^
  --folds 10 ^
  --seed 42 ^
  --n-jobs 1 ^
  --results-dir "%RESULTS%\06_lr_rdkit"
if errorlevel 1 goto :failed

python collect_results.py --results-dir "%RESULTS%\06_lr_rdkit" --latest
if errorlevel 1 goto :failed


rem ============================================================================
rem STEP 7: Logistic Regression fixed baseline, with balancing
rem Saved in results_windows_cmd\07_lr_rdkit_balanced
rem ============================================================================

echo.
echo [STEP 7/9] Balanced Logistic Regression baseline
python Run_experiment.py ^
  -i "%D%.tr_0.2_1.txt" ^
  -p "%D%.te_0.2_1.txt" ^
  --group ED ^
  --features rdkit ^
  --clf lr ^
  --balance ^
  --folds 10 ^
  --seed 42 ^
  --n-jobs 1 ^
  --results-dir "%RESULTS%\07_lr_rdkit_balanced"
if errorlevel 1 goto :failed

python collect_results.py --results-dir "%RESULTS%\07_lr_rdkit_balanced" --latest
if errorlevel 1 goto :failed


rem ============================================================================
rem STEP 8: Low-memory concatenated GVFA test, 18 node features, dimension 100
rem Saved in results_windows_cmd\08_xgb_concat_nf18_dim100_tuned
rem ============================================================================

echo.
echo [STEP 8/9] Tuned XGBoost with concat features, nf18, dimension 100
python Run_experiment.py ^
  -i "%D%.tr_0.2_1.txt" ^
  -p "%D%.te_0.2_1.txt" ^
  --group ED ^
  --features concat ^
  --node-features 18 ^
  --dim 100 ^
  --clf xgb ^
  --tune 50 ^
  --folds 10 ^
  --seed 42 ^
  --n-jobs 1 ^
  --results-dir "%RESULTS%\08_xgb_concat_nf18_dim100_tuned"
if errorlevel 1 goto :failed

python collect_results.py --results-dir "%RESULTS%\08_xgb_concat_nf18_dim100_tuned" --latest
if errorlevel 1 goto :failed


rem ============================================================================
rem STEP 9: Concatenated GVFA test, 18 node features, dimension 500
rem Run only because STEP 8 completed successfully.
rem Saved in results_windows_cmd\09_xgb_concat_nf18_dim500_tuned
rem ============================================================================

echo.
echo [STEP 9/9] Tuned XGBoost with concat features, nf18, dimension 500
python Run_experiment.py ^
  -i "%D%.tr_0.2_1.txt" ^
  -p "%D%.te_0.2_1.txt" ^
  --group ED ^
  --features concat ^
  --node-features 18 ^
  --dim 500 ^
  --clf xgb ^
  --tune 50 ^
  --folds 10 ^
  --seed 42 ^
  --n-jobs 1 ^
  --results-dir "%RESULTS%\09_xgb_concat_nf18_dim500_tuned"
if errorlevel 1 goto :failed

python collect_results.py --results-dir "%RESULTS%\09_xgb_concat_nf18_dim500_tuned" --latest
if errorlevel 1 goto :failed


echo.
echo ============================================================================
echo ALL EXPERIMENTS COMPLETED SUCCESSFULLY
echo Results are under: %RESULTS%
echo Best Optuna parameters are in each tuned folder under configs\ED
echo ============================================================================
exit /b 0


:failed
echo.
echo ============================================================================
echo ERROR: the batch stopped because the command above failed.
echo Fix that error, then run the failed command manually or restart this batch.
echo Existing result and cache files are not deleted.
echo ============================================================================
exit /b 1
