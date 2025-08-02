"""
exgep:
python main.py exgep --geno ./dataset/exgep_data/genotype.csv --phen ./dataset/exgep_data/pheno.csv --soil ./dataset/exgep_data/soil.csv --weather ./dataset/exgep_data/weather.csv --n_traits 1 --target_trait Yield --models_optimize LightGBM --models_assess LightGBM --metric_assess mae,mse,rmse,pcc,r2 --metric_optimise r2 --n_trial 2 --n_splits 2 --write_folder ./exgep_results/ --reload_study --reload_trial --shap_n_train_points 200 --shap_n_test_points 200 --shap_cluster True --shap_model_name LightGBM

autogs：
KFold:
python main.py autogs --cv_type KFold --phen_file ./dataset/autogs_data/trainset/Pheno/ --env_file ./dataset/autogs_data/trainset/Env/ --geno_file ./dataset/autogs_data/trainset/Geno/YI_All.vcf --ref_file ./dataset/docs/maizeRef(ALL).csv --file_names ./dataset/env.txt --models_optimize LightGBM,XGBoost --models_assess LightGBM,XGBoost --metric_assess mae,mse,rmse,pcc,r2 --metric_optimise r2 --n_trial 2  --n_traits 9 --target_trait Yield_Mg_ha --shap_n_train_points 200 --shap_n_test_points 200 --shap_cluster True --shap_model_name LightGBM

LOECV STECV LOESTCV
python main.py autogs --phen_file ./dataset/autogs_data/trainset/Pheno/ --env_file ./dataset/autogs_data/trainset/Env/ --geno_file ./dataset/autogs_data/trainset/Geno/YI_All.vcf --ref_file ./dataset/docs/maizeRef(ALL).csv --file_names ./dataset/env.txt --cv_type STECV --n_traits 9 --target_trait Yield_Mg_ha --base_models KNN,XGBoost,LightGBM --meta_model ridge --is_ensemble --do_optimize --n_trial 2 --metric_assess pcc,rmse --metric_optimise r2 --optimization_objective maximize --write_folder ./autogs_LOECV/

"""

import argparse
import pathlib
import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from autogs.model import RegAutoGS
from autogs.data import datautils as autogs_datautils
from autogs.data.tools.reg_metrics import (mae_score as mae, 
                                           mse_score as mse, 
                                           rmse_score as rmse, 
                                           r2_score as r2,
                                           rmsle_score as rmsle, 
                                           mape_score as mape, 
                                           medae_score as medae, 
                                           pcc_score as pcc)
                                           
from autogs.data.tools.cv_models import STECV, LOECV, LOESTCV
from autogs.model.model_plus import AutoGS

from exgep.model import RegEXGEP
from exgep.data import datautils as exgep_datautils

class Colors:
    AUTOGS_HEADER = '\033[94m'      
    AUTOGS_INFO = '\033[96m'        
    AUTOGS_SUCCESS = '\033[92m'    
    AUTOGS_WARNING = '\033[93m'   
    
    EXGEP_HEADER = '\033[94m'       
    EXGEP_INFO = '\033[96m'         
    EXGEP_SUCCESS = '\033[92m'      
    EXGEP_WARNING = '\033[93m'      
    
    # Common colors
    ERROR = '\033[91m'              
    BOLD = '\033[1m'                
    UNDERLINE = '\033[4m'          
    END = '\033[0m'                

def print_autogs_banner():
    banner = f"""
{Colors.AUTOGS_HEADER}{Colors.BOLD}
----------------------Welcome to AutoGS----------------------

     █████╗ ██╗   ██╗████████╗ ██████╗  ██████╗ ███████╗
    ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗██╔════╝ ██╔════╝
    ███████║██║   ██║   ██║   ██║   ██║██║  ███╗███████╗
    ██╔══██║██║   ██║   ██║   ██║   ██║██║   ██║╚════██║
    ██║  ██║╚██████╔╝   ██║   ╚██████╔╝╚██████╔╝███████║
      ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
{Colors.AUTOGS_HEADER}    Automated Genomic Selection Framework
-------------------------------------------------------------{Colors.BOLD}
"""
    print(banner)

def print_exgep_banner():
    banner = f"""
{Colors.EXGEP_HEADER}{Colors.BOLD}
------------------Welcome to EXGEP------------------

    ███████╗██╗  ██╗ ██████╗ ███████╗██████╗ 
    ██╔════╝╚██╗██╔╝██╔════╝ ██╔════╝██╔══██╗
    █████╗   ╚███╔╝ ██║  ███╗█████╗  ██████╔╝
    ██╔══╝   ██╔██╗ ██║   ██║██╔══╝  ██╔═══╝ 
    ███████╗██╔╝ ██╗╚██████╔╝███████╗██║     
      ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝     
{Colors.EXGEP_HEADER}    Explainable GxE Interaction Prediction Framework
----------------------------------------------------{Colors.BOLD}
"""
    print(banner)

def print_step(step_num, total_steps, description, color=Colors.AUTOGS_INFO):
    print(f"{color}[{step_num}/{total_steps}] {description}...{Colors.END}")

def print_success(message, color=Colors.AUTOGS_SUCCESS):
    print(f"{color}✓ {message}{Colors.END}")

def print_warning(message, color=Colors.AUTOGS_WARNING):
    print(f"{color}⚠ {message}{Colors.END}")

def print_error(message):
    print(f"{Colors.ERROR}✗ {message}{Colors.END}")

def print_info(message, color=Colors.AUTOGS_INFO):
    print(f"{color}ℹ {message}{Colors.END}")

def print_training_start(model_name, cv_type=None):
    if model_name == 'AutoGS':
        color = Colors.AUTOGS_SUCCESS
    else:
        color = Colors.EXGEP_SUCCESS
    
    print(f"\n{color}{Colors.BOLD}✓ TRAINING STARTED...{Colors.END}")
  
    if cv_type:
        print(f"{color}Cross-Validation: {cv_type}{Colors.END}")

def print_training_complete(model_name):
    if model_name == 'AutoGS':
        color = Colors.AUTOGS_SUCCESS
    else:
        color = Colors.EXGEP_SUCCESS
        
    print(f"{color}✓ {model_name} training finished successfully!{Colors.END}")

def print_shap_start(model_name):
    if model_name == 'AutoGS':
        color = Colors.AUTOGS_WARNING
    else:
        color = Colors.EXGEP_WARNING
        
    print(f"{color}Generating SHAP explanations for: {model_name}{Colors.END}")

def print_shap_complete(model_name):
    if model_name == 'AutoGS':
        color = Colors.AUTOGS_INFO
    else:
        color = Colors.EXGEP_INFO
        
    print(f"{color}SHAP analysis for {model_name} finished successfully!{Colors.END}")

def parse_args():
    parser = argparse.ArgumentParser(description="Unified ShapGE CLI")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # KFold/STECV/LOECV/LOESTCV shared parameters)
    autogs_parser = subparsers.add_parser('autogs', help='AutoGS unified')
    autogs_parser.add_argument('--phen_file', type=str, required=True)
    autogs_parser.add_argument('--env_file', type=str, required=True)
    autogs_parser.add_argument('--geno_file', type=str, required=True)
    autogs_parser.add_argument('--ref_file', type=str, required=True)
    autogs_parser.add_argument('--file_names', type=str, required=True, help="Comma separated env names, or path to a txt file if file ends with .txt")
    autogs_parser.add_argument('--cv_type', type=str, default='KFold', choices=['KFold', 'STECV', 'LOECV', 'LOESTCV'])
    autogs_parser.add_argument('--n_traits', type=int, required=True, help='Number of trait columns (after meta columns)')
    autogs_parser.add_argument('--target_trait', type=str, required=False, help='Target trait (phenotype) column name for prediction')
    autogs_parser.add_argument('--metric_assess', type=str, default='mae,mse,rmse,pcc,r2,rmsle,mape,medae')
    autogs_parser.add_argument('--metric_optimise', type=str, default='r2')
    autogs_parser.add_argument('--write_folder', type=str, default='./autogs_results/')
    autogs_parser.add_argument('--n_trial', type=int, default=3)
    
   # AutoGS KFold parameters
    autogs_parser.add_argument('--test_size', type=float, default=0.2)
    autogs_parser.add_argument('--n_splits', type=int, default=2)
    autogs_parser.add_argument('--reload_study', action='store_true')
    autogs_parser.add_argument('--reload_trial', action='store_true')
    autogs_parser.add_argument('--models_optimize', type=str, default='LightGBM,XGBoost')
    autogs_parser.add_argument('--models_assess', type=str, default='LightGBM,XGBoost')
    autogs_parser.add_argument('--early_stopping_rounds', type=int, default=3)
    autogs_parser.add_argument('--random_state', type=int, default=2024)
    autogs_parser.add_argument('--optimization_objective', type=str, default='maximize')

    # AutoGS STECV/LOECV/LOESTCV parameters
    autogs_parser.add_argument('--base_models', type=str, help="Comma separated, e.g. KNN,XGBoost")
    autogs_parser.add_argument('--meta_model', type=str, help="Meta model for stacking")
    autogs_parser.add_argument('--is_ensemble', action='store_true')
    autogs_parser.add_argument('--do_optimize', action='store_true')
    autogs_parser.add_argument('--optuna_sampler', type=str, default='tpe')
    autogs_parser.add_argument('--optuna_pruner', type=str, default='hyperband')
    autogs_parser.add_argument('--ensemble_method', type=str, default='weighted')
    autogs_parser.add_argument('--feature_cols', type=str, help='Comma-separated, or auto-inferred')
    autogs_parser.add_argument('--target_col', type=str, default='EW')
    autogs_parser.add_argument('--env_column', type=str, default='Env')
    autogs_parser.add_argument('--hybrid_column', type=str, default='Hybrid')
    
    # SHAP parameters
    autogs_parser.add_argument('--shap_n_train_points', type=int, default=200, help='Number of training samples or KMeans centers for SHAP background data.')
    autogs_parser.add_argument('--shap_n_test_points', type=int, default=200, help='Number of test samples to explain with SHAP.')
    autogs_parser.add_argument('--shap_cluster', type=lambda x: str(x).lower()=='true', default=True, help='Use KMeans cluster centers for SHAP background data (True/False).')
    autogs_parser.add_argument('--shap_model_name', type=str, default=None, help='Model to explain with SHAP. Use ensemble model name or base model name.')

    # EXGEP
    exgep_parser = subparsers.add_parser('exgep', help='EXGEP ensemble model')
    exgep_parser.add_argument('--geno', type=str, required=True)
    exgep_parser.add_argument('--phen', type=str, required=True)
    exgep_parser.add_argument('--soil', type=str, required=False)
    exgep_parser.add_argument('--weather', type=str, required=False)
    exgep_parser.add_argument('--n_traits', type=int, required=True, help='Number of trait columns (after meta columns)')    
    exgep_parser.add_argument('--target_trait', type=str, required=True, help='Target trait (phenotype) column name for prediction')
    exgep_parser.add_argument('--test_size', type=float, default=0.2)
    exgep_parser.add_argument('--n_splits', type=int, default=2)
    exgep_parser.add_argument('--n_trial', type=int, default=2)
    exgep_parser.add_argument('--reload_study', action='store_true')
    exgep_parser.add_argument('--reload_trial', action='store_true')
    exgep_parser.add_argument('--write_folder', type=str, default='./exgep_results/')
    exgep_parser.add_argument('--models_optimize', type=str, default='XGBoost')
    exgep_parser.add_argument('--models_assess', type=str, default='XGBoost')
    exgep_parser.add_argument('--early_stopping_rounds', type=int, default=3)
    exgep_parser.add_argument('--random_state', type=int, default=2024)
    exgep_parser.add_argument('--metric_assess', type=str, default='mae,mse,rmse,pcc,r2,rmsle,mape,medae')
    exgep_parser.add_argument('--metric_optimise', type=str, default='r2')
    exgep_parser.add_argument('--optimization_objective', type=str, default='maximize')
    
    # SHAP parameters
    exgep_parser.add_argument('--shap_n_train_points', type=int, default=200, help='Number of training samples or KMeans centers for SHAP background data.')
    exgep_parser.add_argument('--shap_n_test_points', type=int, default=200, help='Number of test samples to explain with SHAP.')
    exgep_parser.add_argument('--shap_cluster', type=lambda x: str(x).lower()=='true', default=True, help='Use KMeans cluster centers for SHAP background data (True/False).')
    exgep_parser.add_argument('--shap_model_name', nargs='+', type=str,default=None, help='Model to explain with SHAP. Use ensemble model name or base model name.')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.mode == 'autogs':
        print_autogs_banner()
        
        # file_names = [x.strip() for x in args.file_names.split(',')]
        if args.file_names.endswith('.txt'):
            with open(args.file_names, 'r') as f:
                file_names = [line.strip() for line in f if line.strip()]
        else:
            file_names = [x.strip() for x in args.file_names.split(',')]
            
        print(f"\n{Colors.AUTOGS_INFO}Processing data files...{Colors.END}")
        com_phen_data, com_env_data, dynamic_window_avg, env_transformed_data, \
        gendata, PGE = autogs_datautils.process_data(
            args.phen_file, args.env_file, args.geno_file, args.ref_file, file_names)

        meta_cols = ['Env', 'Hybrid']
        n_traits = int(args.n_traits)
        target_trait = args.target_trait

        meta_end = len(meta_cols)
        trait_cols = list(PGE.columns[meta_end : meta_end + n_traits])

        if not trait_cols:
            print_error("Trait columns not found! Check n_traits or data format.")
            raise ValueError("Trait columns not found! Check n_traits or data format.")
        if not target_trait:
            target_trait = trait_cols[0]
        elif target_trait not in trait_cols:
            print_error(f"Target trait {target_trait} not found in trait columns: {trait_cols}")
            raise ValueError(f"Target trait {target_trait} not found in trait columns: {trait_cols}")

        snp_cols = list(PGE.columns[meta_end + n_traits :])

        selected_cols = meta_cols + [target_trait] + snp_cols
        extracted_columns = PGE.loc[:, selected_cols].dropna().reset_index(drop=True)

        snp = extracted_columns[snp_cols]
        scaler = StandardScaler()
        scaled_snp = scaler.fit_transform(snp)
        X = pd.DataFrame(scaled_snp, columns=snp.columns)

        y = extracted_columns[target_trait]
        y = pd.core.series.Series(y)

        print_info(f"Target trait: {target_trait}", Colors.AUTOGS_INFO)
        print_info(f"Features: {len(snp_cols)} Features", Colors.AUTOGS_INFO)
        print_info(f"Samples: {len(X)} individuals", Colors.AUTOGS_INFO)
        print_info(f"Environments: {len(PGE['Env'].unique())} environments", Colors.AUTOGS_INFO)
        print_success("Data preprocessing completed!", Colors.AUTOGS_SUCCESS)

        metric_dict = {'mae': mae, 'mse': mse, 'rmse': rmse, 'pcc': pcc, 'r2': r2,
                       'rmsle': rmsle, 'mape': mape, 'medae': medae}

        metric_assess_names = [k.strip() for k in args.metric_assess.split(',') if k.strip()]
        metric_optimise_name = args.metric_optimise.strip()

        if args.cv_type == 'KFold':
            metric_assess = [metric_dict[k] for k in metric_assess_names if k in metric_dict]
            metric_optimise = metric_dict.get(metric_optimise_name, r2)
        else:
            metric_assess = metric_assess_names
            metric_optimise = metric_optimise_name

        if args.cv_type == 'KFold':
            print_training_start('AutoGS', args.cv_type)
            
            reg = RegAutoGS(
                y=y,
                X=X,
                test_size=args.test_size,
                n_splits=args.n_splits,
                n_trial=args.n_trial,
                reload_study=args.reload_study,
                reload_trial=args.reload_trial,
                write_folder=args.write_folder,
                metric_optimise=metric_optimise,
                metric_assess=metric_assess,
                optimization_objective=args.optimization_objective,
                models_optimize=[m.strip() for m in args.models_optimize.split(',')],
                models_assess=[m.strip() for m in args.models_assess.split(',')],
                early_stopping_rounds=args.early_stopping_rounds,
                random_state=args.random_state
            )
            reg.train()
            
            print_training_complete('AutoGS')
            print_shap_start('AutoGS')
            
            reg.CalSHAP(
                        n_train_points=args.shap_n_train_points,
                        n_test_points=args.shap_n_test_points,
                        cluster=args.shap_cluster,
                        model_name=args.shap_model_name)
            
            print_shap_complete('AutoGS')
            
        else:
            cv_func_map = {'STECV': STECV, 'LOECV': LOECV, 'LOESTCV': LOESTCV}
            cv_func = cv_func_map[args.cv_type]

            df = pd.concat([
                extracted_columns[meta_cols + [target_trait]].reset_index(drop=True),
                X.reset_index(drop=True)
            ], axis=1)
            feature_cols = [col for col in df.columns if col not in meta_cols + [target_trait]]
            
            print_training_start('AutoGS', args.cv_type)
            
            pipeline = AutoGS(
                base_models=[m.strip() for m in (args.base_models or '').split(',')] if args.base_models else None,
                meta_model=args.meta_model,
                is_ensemble=args.is_ensemble,
                do_optimize=args.do_optimize,
                n_trial=args.n_trial,
                metric_assess=metric_assess,
                metric_optimise=metric_optimise,
                optimization_objective=args.optimization_objective,
                optuna_sampler=None,
                optuna_pruner=None,
                cv_func=cv_func,
                feature_cols=feature_cols,
                target_col=target_trait, 
                env_column=args.env_column,
                hybrid_column=args.hybrid_column,
                write_folder=args.write_folder,
                ensemble_method=args.ensemble_method
            )
            pipeline.fit(df)
            
            print_training_complete('AutoGS')

    elif args.mode == 'exgep':
        print_exgep_banner()
        print(f"\n{Colors.EXGEP_INFO}Processing data files...{Colors.END}")
        print_info(f"Genotype file: {args.geno}", Colors.EXGEP_INFO)
        print_info(f"Phenotype file: {args.phen}", Colors.EXGEP_INFO)
        if args.soil:
            print_info(f"Soil file: {args.soil}", Colors.EXGEP_INFO)
        if args.weather:
            print_info(f"Weather file: {args.weather}", Colors.EXGEP_INFO)
            
        data = exgep_datautils.merge_data(
            genotype_path=args.geno,
            pheno_path=args.phen,
            soil_path=args.soil,
            weather_path=args.weather
        )
        
        meta_cols = ['Env', 'Hybrid']
        n_traits = int(args.n_traits)
        target_trait = args.target_trait

        meta_end = len(meta_cols)
        trait_cols = list(data.columns[meta_end : meta_end + n_traits])

        if not trait_cols:
            print_error("Trait columns not found! Check n_traits or data format.")
            raise ValueError("Trait columns not found! Check n_traits or data format.")
        if target_trait not in trait_cols:
            print_error(f"Target trait {target_trait} not found in trait columns: {trait_cols}")
            raise ValueError(f"Target trait {target_trait} not found in trait columns: {trait_cols}")

        X = pd.DataFrame(data.iloc[:, meta_end + n_traits:])  
        y = data[target_trait]
        y = pd.core.series.Series(y)
        
        print_info(f"Target trait: {target_trait}", Colors.EXGEP_INFO)
        print_info(f"Features: {X.shape[1]} variables", Colors.EXGEP_INFO)
        print_info(f"Samples: {len(X)} individuals", Colors.EXGEP_INFO)
        print_info(f"Environments: {len(data['Env'].unique())} environments", Colors.EXGEP_INFO)
        print_success("Data processing completed!", Colors.EXGEP_SUCCESS)
        
        metric_dict = {'mae': mae, 'mse': mse, 'rmse': rmse, 'pcc': pcc, 'r2': r2,
                       'rmsle': rmsle, 'mape': mape, 'medae': medae}
                       
        metric_assess = [metric_dict[k.strip()] for k in args.metric_assess.split(',') if k.strip() in metric_dict]
        metric_optimise = metric_dict.get(args.metric_optimise.strip(), r2)
        
        print_training_start('EXGEP')
        
        reg = RegEXGEP(
                       y=y,
                       X=X,
                       test_size=args.test_size,
                       n_splits=args.n_splits,
                       n_trial=args.n_trial,
                       reload_study=args.reload_study,
                       reload_trial=args.reload_trial,
                       write_folder=args.write_folder,
                       metric_optimise=metric_optimise,
                       metric_assess=metric_assess,
                       optimization_objective=args.optimization_objective,
                       models_optimize=[m.strip() for m in args.models_optimize.split(',')],
                       models_assess=[m.strip() for m in args.models_assess.split(',')],
                       early_stopping_rounds=args.early_stopping_rounds,
                       random_state=args.random_state
        )
        reg.train()
        
        print_training_complete('EXGEP')
        print_shap_start('EXGEP')
        
        reg.CalSHAP(
                    n_train_points=args.shap_n_train_points,
                    n_test_points=args.shap_n_test_points,
                    cluster=args.shap_cluster,
                    model_name=args.shap_model_name)
        
        print_shap_complete('EXGEP')

    print_success("All processes completed successfully! ", Colors.AUTOGS_SUCCESS)

if __name__ == '__main__':
    main()