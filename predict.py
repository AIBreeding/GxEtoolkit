"""
exgep:
python predict.py exgep --geno ./dataset/exgep_data/genotype.csv --phen ./dataset/exgep_data/pheno.csv --soil ./dataset/exgep_data/soil.csv --weather ./dataset/exgep_data/weather.csv --n_traits 1 --target_trait Yield --metric_assess pcc,rmse,mse,mae,r2 --save_path ./id_exgep_predict --model_path ./exgep_results/estimators/LightGBM_best_model.joblib

autogs：
python predict.py autogs --phen_file ./dataset/autogs_data/trainset/Pheno/ --env_file ./dataset/autogs_data/trainset/Env/ --geno_file ./dataset/autogs_data/trainset/Geno/YI_All.vcf --ref_file ./dataset/docs/maizeRef(ALL).csv --file_names ./dataset/env.txt --n_traits 9 --target_trait Yield_Mg_ha --metric_assess pcc,mse,rmse,mae --save_path ./autogs_pre/ --model_path ./autogs_results/AutoGS_model.joblib

"""
import argparse
import pathlib
import os
import sys
import joblib
from collections import OrderedDict
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import numpy as np

project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from autogs.data import datautils as autogs_datautils
from exgep.data import datautils as exgep_datautils
from autogs.data.tools.reg_metrics import (
    mae_score as mae,
    mse_score as mse,
    rmse_score as rmse,
    r2_score as r2,
    rmsle_score as rmsle,
    mape_score as mape,
    medae_score as medae,
    pcc_score as pcc
)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Unified ShapGE CLI")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # AutoGS KFold/STECV/LOECV/LOESTCV
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
    autogs_parser.add_argument('--save_path', type=str, default='./autogs_results/')
    autogs_parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (.joblib)')

    # EXGEP
    exgep_parser = subparsers.add_parser('exgep', help='EXGEP ensemble model')
    exgep_parser.add_argument('--geno', type=str, required=True)
    exgep_parser.add_argument('--phen', type=str, required=True)
    exgep_parser.add_argument('--soil', type=str, required=False)
    exgep_parser.add_argument('--weather', type=str, required=False)
    exgep_parser.add_argument('--n_traits', type=int, required=True, help='Number of trait columns (after meta columns)')
    exgep_parser.add_argument('--target_trait', type=str, required=True, help='Target trait (phenotype) column name for prediction')
    exgep_parser.add_argument('--metric_assess', type=str, default='mae,mse,rmse,pcc,r2,rmsle,mape,medae')
    exgep_parser.add_argument('--save_path', type=str, default='./exgep_results/')
    exgep_parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (.joblib)')

    return parser.parse_args()

def evaluate_performance(pred_value, target_trait, metric_dict, metric_assess_names, save_path, mode='autogs'):
    color_info = Colors.AUTOGS_INFO if mode == 'autogs' else Colors.EXGEP_INFO
    color_success = Colors.AUTOGS_SUCCESS if mode == 'autogs' else Colors.EXGEP_SUCCESS
    color_warning = Colors.AUTOGS_WARNING if mode == 'autogs' else Colors.EXGEP_WARNING
    
    print_step(4, 4, f"Evaluating model performance for {target_trait}", color_info)
    
    results = []
    sample_stat = []

    environments = pred_value['Env'].unique()
    print_info(f"Found {len(environments)} environments: {', '.join(environments)}", color_info)
    
    for env in environments:
        current_data = pred_value[pred_value['Env'] == env].copy()
        sample_count = len(current_data)
        sample_stat.append({'Env': env, 'n_samples': sample_count})

        if sample_count < 2:
            print_warning(f"Env {env}: less than 2 samples, skip", color_warning)
            continue

        current_data = current_data.dropna(subset=[target_trait, 'pred'])
        current_data[target_trait] = current_data[target_trait].astype(float)
        current_data['pred'] = current_data['pred'].astype(float)

        if current_data.empty:
            print_warning(f"Env {env}: all values NaN after dropna, skip", color_warning)
            continue

        print_info(f"Processing environment: {env} ({sample_count} samples)", color_info)
        
        row = OrderedDict()
        row['Env'] = env
        for metric_name in metric_assess_names:
            try:
                if metric_name == "pcc":
                    if current_data[target_trait].nunique() == 1 or current_data['pred'].nunique() == 1:
                        value = np.nan
                    else:
                        value, _ = pearsonr(current_data[target_trait], current_data['pred'])
                else:
                    value = metric_dict[metric_name](current_data[target_trait], current_data['pred'])
            except Exception as e:
                print_error(f"Metric {metric_name} failed for env {env}: {e}")
                value = np.nan
            row[metric_name] = value
        results.append(row)

    # Global evaluation
    global_data = pred_value.dropna(subset=[target_trait, 'pred']).copy()
    global_data[target_trait] = global_data[target_trait].astype(float)
    global_data['pred'] = global_data['pred'].astype(float)
    
    print_info(f"Computing global metrics across all environments ({len(global_data)} samples)", color_info)
    
    global_row = OrderedDict()
    global_row['Env'] = 'Global'
    for metric_name in metric_assess_names:
        try:
            if metric_name == "pcc":
                if global_data[target_trait].nunique() == 1 or global_data['pred'].nunique() == 1:
                    value = np.nan
                else:
                    value, _ = pearsonr(global_data[target_trait], global_data['pred'])
            else:
                value = metric_dict[metric_name](global_data[target_trait], global_data['pred'])
        except Exception as e:
            print_error(f"Metric {metric_name} failed for Global: {e}")
            value = np.nan
        global_row[metric_name] = value
    results.append(global_row)

    # Average across environments
    if len(results) > 1:
        env_results = [r for r in results if r['Env'] != 'Global']
        if env_results:
            avg_row = OrderedDict()
            avg_row['Env'] = 'Average'
            for metric_name in metric_assess_names:
                valid_metrics = [r[metric_name] for r in env_results if not pd.isnull(r[metric_name])]
                if valid_metrics:
                    avg_value = sum(valid_metrics) / len(valid_metrics)
                else:
                    avg_value = np.nan
                avg_row[metric_name] = avg_value
            results.append(avg_row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_path, "results_df.csv"), index=False)
    
    print_success(f"Results saved to {os.path.join(save_path, 'results_df.csv')}", color_success)
    
    # Display summary of results
    # print(f"\n{color_info} Performance Summary:{Colors.END}")
    # print(f"{color_info}{'='*50}{Colors.END}")
    # for _, row in results_df.iterrows():
    #     env_name = row['Env']
    #     if 'pcc' in row and not pd.isnull(row['pcc']):
    #         pcc_val = row['pcc']
    #         print(f"{color_info}{env_name:>10}: PCC = {pcc_val:.4f}{Colors.END}")

def main():
    args = parse_args()
    
    if args.mode == 'autogs':
        print_autogs_banner()
        print_step(1, 4, "Loading and processing data files", Colors.AUTOGS_INFO)
        print_info(f"Phenotype file: {args.phen_file}", Colors.AUTOGS_INFO)
        print_info(f"Environment file: {args.env_file}", Colors.AUTOGS_INFO)
        print_info(f"Genotype file: {args.geno_file}", Colors.AUTOGS_INFO)
        print_info(f"Reference file: {args.ref_file}", Colors.AUTOGS_INFO)
        
        if args.file_names.endswith('.txt'):
            with open(args.file_names, 'r') as f:
                file_names = [line.strip() for line in f if line.strip()]
            print_info(f"Environment names loaded from file: {len(file_names)} environments", Colors.AUTOGS_INFO)
        else:
            file_names = [x.strip() for x in args.file_names.split(',')]
            print_info(f"Environment names: {', '.join(file_names)}", Colors.AUTOGS_INFO)

        try:
            com_phen_data, com_env_data, dynamic_window_avg, env_transformed_data, \
            gendata, PGE = autogs_datautils.process_data(
                args.phen_file, args.env_file, args.geno_file, args.ref_file, file_names)
            print_success("Data processing completed successfully", Colors.AUTOGS_SUCCESS)
        except Exception as e:
            print_error(f"Data processing failed: {e}")
            return

        print_step(2, 4, "Preparing features and target variables", Colors.AUTOGS_INFO)
        
        meta_cols = ['Env', 'Hybrid']
        n_traits = int(args.n_traits)
        target_trait = args.target_trait

        meta_end = len(meta_cols)
        trait_cols = list(PGE.columns[meta_end : meta_end + n_traits])

        if not trait_cols:
            print_error("Trait columns not found! Check n_traits or data format.")
            return
            
        if not target_trait:
            target_trait = trait_cols[0]
            print_info(f"Using default target trait: {target_trait}", Colors.AUTOGS_INFO)
        elif target_trait not in trait_cols:
            print_error(f"Target trait {target_trait} not found in trait columns: {trait_cols}")
            return
        else:
            print_info(f"Target trait: {target_trait}", Colors.AUTOGS_INFO)

        snp_cols = list(PGE.columns[meta_end + n_traits :])
        print_info(f"Number of SNP features: {len(snp_cols)}", Colors.AUTOGS_INFO)

        selected_cols = meta_cols + [target_trait] + snp_cols
        extracted_columns = PGE.loc[:, selected_cols].dropna().reset_index(drop=True)
        print_info(f"Dataset shape after filtering: {extracted_columns.shape}", Colors.AUTOGS_INFO)

        snp = extracted_columns[snp_cols]
        scaler = StandardScaler()
        scaled_snp = scaler.fit_transform(snp)
        X = pd.DataFrame(scaled_snp, columns=snp.columns)
        print_success("Feature scaling completed", Colors.AUTOGS_SUCCESS)

        y = PGE[['Env','Hybrid',target_trait]].reset_index(drop=True)

        metric_dict = {'mae': mae, 'mse': mse, 'rmse': rmse, 'pcc': pcc, 'r2': r2,
                       'rmsle': rmsle, 'mape': mape, 'medae': medae}
        metric_assess_names = [k.strip() for k in args.metric_assess.split(',') if k.strip()]
        print_info(f"Evaluation metrics: {', '.join(metric_assess_names)}", Colors.AUTOGS_INFO)

        os.makedirs(args.save_path, exist_ok=True)
        print_info(f"Results will be saved to: {args.save_path}", Colors.AUTOGS_INFO)

        print_step(3, 4, "Loading model and making predictions", Colors.AUTOGS_INFO)
        
        try:
            model = joblib.load(args.model_path)
            print_success(f"Model loaded from: {args.model_path}", Colors.AUTOGS_SUCCESS)
        except Exception as e:
            print_error(f"Failed to load model: {e}")
            return

        try:
            pred = pd.DataFrame(model.predict(X), columns=['pred'])
            pred = pred.reset_index(drop=True)
            pred_value = pd.concat([y, pred], axis=1)
            pred_value.to_csv(os.path.join(args.save_path, "pred_value.csv"), index=False)
            print_success(f"Predictions saved to: {os.path.join(args.save_path, 'pred_value.csv')}", Colors.AUTOGS_SUCCESS)
        except Exception as e:
            print_error(f"Prediction failed: {e}")
            return

        evaluate_performance(pred_value, target_trait, metric_dict, metric_assess_names, args.save_path, 'autogs')
        
        print(f"\n{Colors.AUTOGS_SUCCESS}{Colors.BOLD}✓ AutoGS prediction completed successfully! {Colors.END}")

    elif args.mode == 'exgep':
        print_exgep_banner()
        print_step(1, 4, "Loading and merging data files", Colors.EXGEP_INFO)
        print_info(f"Genotype file: {args.geno}", Colors.EXGEP_INFO)
        print_info(f"Phenotype file: {args.phen}", Colors.EXGEP_INFO)
        if args.soil:
            print_info(f"Soil file: {args.soil}", Colors.EXGEP_INFO)
        if args.weather:
            print_info(f"Weather file: {args.weather}", Colors.EXGEP_INFO)
        
        try:
            data = exgep_datautils.merge_data(
                genotype_path=args.geno,
                pheno_path=args.phen,
                soil_path=args.soil,
                weather_path=args.weather
            )
            print_success("Data merging completed successfully", Colors.EXGEP_SUCCESS)
            print_info(f"Merged dataset shape: {data.shape}", Colors.EXGEP_INFO)
        except Exception as e:
            print_error(f"Data merging failed: {e}")
            return
            
        print_step(2, 4, "Preparing features and target variables", Colors.EXGEP_INFO)
        
        meta_cols = ['Env', 'Hybrid']
        n_traits = int(args.n_traits)
        target_trait = args.target_trait

        meta_end = len(meta_cols)
        trait_cols = list(data.columns[meta_end : meta_end + n_traits])

        if not trait_cols:
            print_error("Trait columns not found! Check n_traits or data format.")
            return
            
        if target_trait not in trait_cols:
            print_error(f"Target trait {target_trait} not found in trait columns: {trait_cols}")
            return
        
        print_info(f"Target trait: {target_trait}", Colors.EXGEP_INFO)
        
        X = pd.DataFrame(data.iloc[:, meta_end + n_traits:])
        print_info(f"Number of features: {X.shape[1]}", Colors.EXGEP_INFO)

        y = data[['Env', 'Hybrid', target_trait]].reset_index(drop=True)
        print_info(f"Number of samples: {len(y)}", Colors.EXGEP_INFO)
        
        os.makedirs(args.save_path, exist_ok=True)
        print_info(f"Results will be saved to: {args.save_path}", Colors.EXGEP_INFO)

        print_step(3, 4, "Loading model and making predictions", Colors.EXGEP_INFO)
        
        try:
            model = joblib.load(args.model_path)
            print_success(f"Model loaded from: {args.model_path}", Colors.EXGEP_SUCCESS)
        except Exception as e:
            print_error(f"Failed to load model: {e}")
            return

        try:
            pred = pd.DataFrame(model.predict(X), columns=['pred'])
            pred = pred.reset_index(drop=True)
            pred_value = pd.concat([y, pred], axis=1)
            pred_value.to_csv(os.path.join(args.save_path, "pred_value.csv"), index=False)
            print_success(f"Predictions saved to: {os.path.join(args.save_path, 'pred_value.csv')}", Colors.EXGEP_SUCCESS)
        except Exception as e:
            print_error(f"Prediction failed: {e}")
            return

        metric_dict = {'mae': mae, 'mse': mse, 'rmse': rmse, 'pcc': pcc, 'r2': r2,
                       'rmsle': rmsle, 'mape': mape, 'medae': medae}
        metric_assess_names = [k.strip() for k in args.metric_assess.split(',') if k.strip()]
        print_info(f"Evaluation metrics: {', '.join(metric_assess_names)}", Colors.EXGEP_INFO)

        evaluate_performance(pred_value, target_trait, metric_dict, metric_assess_names, args.save_path, 'exgep')
        
        print(f"\n{Colors.EXGEP_SUCCESS}{Colors.BOLD}✓ EXGEP prediction completed successfully!{Colors.END}")

if __name__ == '__main__':
    main()