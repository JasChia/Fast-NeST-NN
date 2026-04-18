#!/usr/bin/env python3
"""
Statistical analysis of model metrics across drugs for shuffled vs unshuffled experiments.
Performs significance testing and creates comprehensive results table.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon
import warnings
warnings.filterwarnings('ignore')

_HERE = Path(__file__).resolve().parent
_COMBINED_METRICS = _HERE.parent / "nest_vnn" / "combined_model_metrics.csv"

def load_and_prepare_data():
    """Load the combined metrics data and prepare for analysis."""
    df = pd.read_csv(_COMBINED_METRICS)
    
    # Convert Drug_ID to string for consistency
    df['Drug_ID'] = df['Drug_ID'].astype(str)
    
    # Remove rows with NaN values in Validation or Test columns
    df = df.dropna(subset=['Validation', 'Test'])
    
    print(f"Loaded data with {len(df)} rows")
    print(f"Drugs: {sorted(df['Drug_ID'].unique())}")
    print(f"Metrics: {df['Metric'].unique()}")
    
    return df

def perform_statistical_tests(df):
    """Perform statistical tests for each drug and metric combination."""
    results = []
    
    # Get unique drugs and metrics
    drugs = sorted(df['Drug_ID'].unique())
    metrics = df['Metric'].unique()
    
    for drug in drugs:
        print(f"\nAnalyzing Drug {drug}...")
        
        for metric in metrics:
            # Filter data for this drug and metric
            drug_data = df[(df['Drug_ID'] == drug) & (df['Metric'] == metric)]
            
            if len(drug_data) == 0:
                continue
                
            # Separate shuffled and unshuffled data
            shuffled_val = drug_data[drug_data['Shuffle_Status'] == 'shuffled']['Validation'].values
            unshuffled_val = drug_data[drug_data['Shuffle_Status'] == 'unshuffled']['Validation'].values
            
            shuffled_test = drug_data[drug_data['Shuffle_Status'] == 'shuffled']['Test'].values
            unshuffled_test = drug_data[drug_data['Shuffle_Status'] == 'unshuffled']['Test'].values
            
            # Skip if we don't have enough data
            if len(shuffled_val) < 2 or len(unshuffled_val) < 2:
                continue
            
            # Statistical tests for Validation data
            try:
                # T-test
                t_stat_val, t_p_val = ttest_ind(shuffled_val, unshuffled_val)
                
                # Mann-Whitney U test (non-parametric)
                u_stat_val, u_p_val = mannwhitneyu(shuffled_val, unshuffled_val, alternative='two-sided')
                
                # Effect size (Cohen's d)
                pooled_std_val = np.sqrt(((len(shuffled_val) - 1) * np.var(shuffled_val, ddof=1) + 
                                        (len(unshuffled_val) - 1) * np.var(unshuffled_val, ddof=1)) / 
                                       (len(shuffled_val) + len(unshuffled_val) - 2))
                cohens_d_val = (np.mean(shuffled_val) - np.mean(unshuffled_val)) / pooled_std_val
                
            except:
                t_stat_val = t_p_val = u_stat_val = u_p_val = cohens_d_val = np.nan
            
            # Statistical tests for Test data
            try:
                # T-test
                t_stat_test, t_p_test = ttest_ind(shuffled_test, unshuffled_test)
                
                # Mann-Whitney U test (non-parametric)
                u_stat_test, u_p_test = mannwhitneyu(shuffled_test, unshuffled_test, alternative='two-sided')
                
                # Effect size (Cohen's d)
                pooled_std_test = np.sqrt(((len(shuffled_test) - 1) * np.var(shuffled_test, ddof=1) + 
                                         (len(unshuffled_test) - 1) * np.var(unshuffled_test, ddof=1)) / 
                                        (len(shuffled_test) + len(unshuffled_test) - 2))
                cohens_d_test = (np.mean(shuffled_test) - np.mean(unshuffled_test)) / pooled_std_test
                
            except:
                t_stat_test = t_p_test = u_stat_test = u_p_test = cohens_d_test = np.nan
            
            # Calculate means and standard deviations
            shuffled_val_mean = np.mean(shuffled_val)
            unshuffled_val_mean = np.mean(unshuffled_val)
            shuffled_test_mean = np.mean(shuffled_test)
            unshuffled_test_mean = np.mean(unshuffled_test)
            
            # Store results
            results.append({
                'Drug_ID': drug,
                'Metric': metric,
                'Shuffled_Val_Mean': shuffled_val_mean,
                'Unshuffled_Val_Mean': unshuffled_val_mean,
                'Val_Mean_Diff': shuffled_val_mean - unshuffled_val_mean,
                'Shuffled_Test_Mean': shuffled_test_mean,
                'Unshuffled_Test_Mean': unshuffled_test_mean,
                'Test_Mean_Diff': shuffled_test_mean - unshuffled_test_mean,
                'Val_T_Stat': t_stat_val,
                'Val_T_P_Value': t_p_val,
                'Val_MW_U_Stat': u_stat_val,
                'Val_MW_P_Value': u_p_val,
                'Val_Cohens_D': cohens_d_val,
                'Test_T_Stat': t_stat_test,
                'Test_T_P_Value': t_p_test,
                'Test_MW_U_Stat': u_stat_test,
                'Test_MW_P_Value': u_p_test,
                'Test_Cohens_D': cohens_d_test,
                'N_Shuffled': len(shuffled_val),
                'N_Unshuffled': len(unshuffled_val)
            })
    
    return pd.DataFrame(results)

def create_summary_table(results_df):
    """Create a summary table with significance indicators."""
    summary_results = []
    
    for _, row in results_df.iterrows():
        # Determine significance for Validation
        val_significant_t = row['Val_T_P_Value'] < 0.05 if not pd.isna(row['Val_T_P_Value']) else False
        val_significant_mw = row['Val_MW_P_Value'] < 0.05 if not pd.isna(row['Val_MW_P_Value']) else False
        
        # Determine significance for Test
        test_significant_t = row['Test_T_P_Value'] < 0.05 if not pd.isna(row['Test_T_P_Value']) else False
        test_significant_mw = row['Test_MW_P_Value'] < 0.05 if not pd.isna(row['Test_MW_P_Value']) else False
        
        # Effect size interpretation
        val_effect_size = "Large" if abs(row['Val_Cohens_D']) > 0.8 else "Medium" if abs(row['Val_Cohens_D']) > 0.5 else "Small" if not pd.isna(row['Val_Cohens_D']) else "N/A"
        test_effect_size = "Large" if abs(row['Test_Cohens_D']) > 0.8 else "Medium" if abs(row['Test_Cohens_D']) > 0.5 else "Small" if not pd.isna(row['Test_Cohens_D']) else "N/A"
        
        summary_results.append({
            'Drug_ID': row['Drug_ID'],
            'Metric': row['Metric'],
            'Val_Shuffled_Mean': f"{row['Shuffled_Val_Mean']:.4f}",
            'Val_Unshuffled_Mean': f"{row['Unshuffled_Val_Mean']:.4f}",
            'Val_Difference': f"{row['Val_Mean_Diff']:.4f}",
            'Val_Significant_T': "Yes" if val_significant_t else "No",
            'Val_Significant_MW': "Yes" if val_significant_mw else "No",
            'Val_P_Value_T': f"{row['Val_T_P_Value']:.4f}" if not pd.isna(row['Val_T_P_Value']) else "N/A",
            'Val_P_Value_MW': f"{row['Val_MW_P_Value']:.4f}" if not pd.isna(row['Val_MW_P_Value']) else "N/A",
            'Val_Effect_Size': val_effect_size,
            'Test_Shuffled_Mean': f"{row['Shuffled_Test_Mean']:.4f}",
            'Test_Unshuffled_Mean': f"{row['Unshuffled_Test_Mean']:.4f}",
            'Test_Difference': f"{row['Test_Mean_Diff']:.4f}",
            'Test_Significant_T': "Yes" if test_significant_t else "No",
            'Test_Significant_MW': "Yes" if test_significant_mw else "No",
            'Test_P_Value_T': f"{row['Test_T_P_Value']:.4f}" if not pd.isna(row['Test_T_P_Value']) else "N/A",
            'Test_P_Value_MW': f"{row['Test_MW_P_Value']:.4f}" if not pd.isna(row['Test_MW_P_Value']) else "N/A",
            'Test_Effect_Size': test_effect_size,
            'N_Shuffled': row['N_Shuffled'],
            'N_Unshuffled': row['N_Unshuffled']
        })
    
    return pd.DataFrame(summary_results)

def main():
    """Main analysis function."""
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    print("\nPerforming statistical tests...")
    results_df = perform_statistical_tests(df)
    
    print("\nCreating summary table...")
    summary_df = create_summary_table(results_df)
    
    # Save detailed results
    out_det = _HERE / "detailed_statistical_results.csv"
    out_sum = _HERE / "statistical_summary_table.csv"
    results_df.to_csv(out_det, index=False)
    print(f"Detailed results saved to: {out_det}")
    
    # Save summary table
    summary_df.to_csv(out_sum, index=False)
    print(f"Summary table saved to: {out_sum}")
    
    # Display summary
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE SUMMARY")
    print("="*80)
    
    # Count significant results
    val_sig_t = len(summary_df[summary_df['Val_Significant_T'] == 'Yes'])
    val_sig_mw = len(summary_df[summary_df['Val_Significant_MW'] == 'Yes'])
    test_sig_t = len(summary_df[summary_df['Test_Significant_T'] == 'Yes'])
    test_sig_mw = len(summary_df[summary_df['Test_Significant_MW'] == 'Yes'])
    
    print(f"Validation Data - Significant differences (T-test): {val_sig_t}/{len(summary_df)}")
    print(f"Validation Data - Significant differences (Mann-Whitney): {val_sig_mw}/{len(summary_df)}")
    print(f"Test Data - Significant differences (T-test): {test_sig_t}/{len(summary_df)}")
    print(f"Test Data - Significant differences (Mann-Whitney): {test_sig_mw}/{len(summary_df)}")
    
    # Show first few rows of summary
    print(f"\nFirst 10 rows of summary table:")
    print(summary_df.head(10).to_string(index=False))
    
    return summary_df, results_df

if __name__ == "__main__":
    summary_df, results_df = main()
