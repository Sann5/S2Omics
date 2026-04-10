import math
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

PROJECT_ROOT = next(
    (candidate for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (candidate / 's2omics').exists()),
    Path.cwd().resolve(),
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from s2omics.s1_utils import load_pickle


try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)


def _find_column(df, candidates):
    lower_map = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match
    return None


def _normalize_event_binary(series):
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().sum() > 0:
        binary = numeric.round().clip(lower=0, upper=1)
        return binary.where(numeric.notna())

    cleaned = series.astype(str).str.strip().str.lower()
    event_tokens = {'1', 'true', 'yes', 'dead', 'death', 'event', 'deceased', 'progression', 'progressed'}
    censor_tokens = {'0', 'false', 'no', 'alive', 'censored', 'censor', 'no progression'}

    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[cleaned.isin(event_tokens)] = 1.0
    out[cleaned.isin(censor_tokens)] = 0.0
    return out


def _normalize_save_folder(save_folder):
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    (save_folder / 'image_files').mkdir(parents=True, exist_ok=True)
    (save_folder / 'pickle_files').mkdir(parents=True, exist_ok=True)
    return str(save_folder) + '/'


def discover_save_folders(root_dir, sample_names=None, require_cluster_image=True):
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f'Root directory does not exist: {root}')

    if sample_names:
        candidates = [root / name for name in sample_names]
    else:
        candidates = [child for child in sorted(root.iterdir()) if child.is_dir()]

    save_folders = []
    for candidate in candidates:
        if (candidate / 'pickle_files').exists():
            save_folder = candidate
        elif (candidate / 'S2Omics_output').exists():
            save_folder = candidate / 'S2Omics_output'
        else:
            continue

        if require_cluster_image and not (save_folder / 'pickle_files' / 'cluster_image.pickle').exists():
            continue

        save_folders.append(str(save_folder.resolve()))

    if not save_folders:
        raise ValueError(
            f'No save folders found under {root}. '
            'Expected either sample folders containing S2Omics_output/pickle_files '
            'or direct S2Omics_output folders with cluster_image.pickle.'
        )

    return save_folders


def summarize_clusters(save_folder_list, patient_ids, output_path, patch_size=16, pixel_size=0.5):
    if len(save_folder_list) != len(patient_ids):
        raise ValueError(
            f'save_folder_list ({len(save_folder_list)}) and patient_ids '
            f'({len(patient_ids)}) must have the same length.'
        )

    superpixel_area_um2 = (patch_size * pixel_size) ** 2
    records = []

    for save_folder, pid in zip(save_folder_list, patient_ids):
        save_folder = _normalize_save_folder(save_folder)
        pickle_folder = save_folder + 'pickle_files/'

        cluster_image = load_pickle(pickle_folder + 'cluster_image.pickle')

        tissue_mask = cluster_image >= 0
        n_tissue = int(tissue_mask.sum())
        tissue_area_um2 = n_tissue * superpixel_area_um2

        cluster_labels = cluster_image[tissue_mask].astype(int)
        unique_clusters = np.unique(cluster_labels)

        row = {
            'patient_id': pid,
            'tissue_area_um2': tissue_area_um2,
            'n_tissue_superpixels': n_tissue,
        }

        for k in unique_clusters:
            count_k = int((cluster_labels == k).sum())
            row[f'freq_cluster_{k}'] = count_k / n_tissue if n_tissue > 0 else 0.0
            row[f'density_cluster_{k}'] = count_k / tissue_area_um2 if tissue_area_um2 > 0 else 0.0

        records.append(row)

    df = pd.DataFrame(records).fillna(0.0)
    meta_cols = ['patient_id', 'tissue_area_um2', 'n_tissue_superpixels']
    freq_cols = sorted([c for c in df.columns if c.startswith('freq_cluster_')])
    density_cols = sorted([c for c in df.columns if c.startswith('density_cluster_')])
    df = df[meta_cols + freq_cols + density_cols]

    output_dir = os.path.dirname(str(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f'Cluster summary saved to: {output_path}')
    print(f'  Patients: {len(df)}')
    print(f'  Clusters: {len(freq_cols)}')
    return df


def summarize_clusters_from_root(root_dir, output_path, sample_names=None, patch_size=16, pixel_size=0.5):
    save_folder_list = discover_save_folders(root_dir, sample_names=sample_names)
    patient_ids = [Path(save_folder).parent.name for save_folder in save_folder_list]
    return summarize_clusters(
        save_folder_list=save_folder_list,
        patient_ids=patient_ids,
        output_path=output_path,
        patch_size=patch_size,
        pixel_size=pixel_size,
    )


def _normalize_patient_id(value):
    if pd.isna(value):
        return ''
    return str(value).strip().strip('"').strip("'")


def _compact_patient_id(value):
    return re.sub(r'[^A-Za-z0-9]+', '', _normalize_patient_id(value).lower())


def _match_score(left_id, right_id):
    left_compact = _compact_patient_id(left_id)
    right_compact = _compact_patient_id(right_id)
    if not left_compact or not right_compact:
        return 0.0
    if left_compact in right_compact or right_compact in left_compact:
        return 1.0
    return SequenceMatcher(None, left_compact, right_compact).ratio()


def _best_match(source_id, candidate_ids, min_score=0.75):
    best_candidate = None
    best_score = 0.0
    for candidate_id in candidate_ids:
        score = _match_score(source_id, candidate_id)
        if score > best_score:
            best_candidate = candidate_id
            best_score = score
    if best_score < min_score:
        return None, best_score
    return best_candidate, best_score


def merge_cluster_with_clinical(clinical_path, cluster_summary_path, output_path, min_score=0.75):
    clinical_df = pd.read_csv(clinical_path, sep='\t', quotechar='"', dtype=str)
    clinical_df.columns = [col.strip().strip('"') for col in clinical_df.columns]
    clinical_df['clinical_patient_id'] = clinical_df['patient_id'].map(_normalize_patient_id)
    clinical_df = clinical_df.drop(columns=['patient_id'])

    cluster_df = pd.read_csv(cluster_summary_path)
    cluster_df['cluster_patient_id'] = cluster_df['patient_id'].map(_normalize_patient_id)
    cluster_df = cluster_df.drop(columns=['patient_id'])

    clinical_ids = clinical_df['clinical_patient_id'].tolist()
    match_records = []
    for cluster_id in cluster_df['cluster_patient_id']:
        matched_id, score = _best_match(cluster_id, clinical_ids, min_score=min_score)
        match_records.append((matched_id, score))

    cluster_df['matched_clinical_patient_id'] = [matched_id for matched_id, _ in match_records]
    cluster_df['match_score'] = [score for _, score in match_records]
    cluster_df['join_patient_id'] = cluster_df['matched_clinical_patient_id'].fillna(cluster_df['cluster_patient_id'])
    clinical_df['join_patient_id'] = clinical_df['clinical_patient_id']

    merged_df = cluster_df.merge(
        clinical_df,
        on='join_patient_id',
        how='outer',
        suffixes=('_cluster', '_clinical'),
        indicator=True,
    )

    merged_df['patient_id'] = merged_df['cluster_patient_id'].fillna(merged_df['clinical_patient_id'])
    merged_df = merged_df.drop(columns=['join_patient_id'])

    merged_df = merged_df[[
        'patient_id',
        'cluster_patient_id',
        'clinical_patient_id',
        'matched_clinical_patient_id',
        'match_score',
        '_merge',
    ] + [col for col in merged_df.columns if col not in {
        'patient_id',
        'cluster_patient_id',
        'clinical_patient_id',
        'matched_clinical_patient_id',
        'match_score',
        '_merge',
    }]]

    output_dir = os.path.dirname(str(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    merged_df.to_csv(output_path, index=False, na_rep='NA')

    print(f'Clinical rows: {len(clinical_df)}')
    print(f'Cluster summary rows: {len(cluster_df)}')
    print(f'Merged rows: {len(merged_df)}')
    print(f'Cluster rows matched to clinical data: {merged_df["cluster_patient_id"].notna().sum()}')
    print(f'Clinical rows matched to cluster data: {merged_df["clinical_patient_id"].notna().sum()}')
    print(f'Unmatched cluster rows: {merged_df[(merged_df["_merge"] == "left_only")].shape[0]}')
    print(f'Unmatched clinical rows: {merged_df[(merged_df["_merge"] == "right_only")].shape[0]}')
    print(f'Merged CSV saved to: {output_path}')

    return merged_df


def _plot_km(time_values, event_values, label, ax):
    sf = SurvfuncRight(time_values, event_values)
    ax.step(sf.surv_times, sf.surv_prob, where='post', label=label)


def _plot_grouped_km_by_treatment(df, time_col, event_col, treatment_col, title, min_group_n=10):
    if treatment_col is None:
        print(f'[{title}] Skipped: Broad_treatment column not found.')
        return None

    km_df = df[[time_col, event_col, treatment_col]].copy()
    km_df[time_col] = pd.to_numeric(km_df[time_col], errors='coerce')
    km_df[event_col] = _normalize_event_binary(km_df[event_col])
    km_df[treatment_col] = km_df[treatment_col].astype(str).str.strip()

    invalid_labels = {'', 'na', 'nan', 'none', 'unknown'}
    km_df = km_df.dropna(subset=[time_col, event_col, treatment_col])
    km_df = km_df[km_df[time_col] >= 0]
    km_df = km_df[~km_df[treatment_col].str.lower().isin(invalid_labels)]

    counts = km_df[treatment_col].value_counts()
    valid_groups = counts[counts >= min_group_n].index.tolist()

    if len(valid_groups) < 2:
        print(f'[{title}] Skipped: fewer than 2 Broad_treatment groups with at least {min_group_n} samples.')
        if not counts.empty:
            print('Observed groups and counts:')
            print(counts.to_string())
        return None

    km_df = km_df[km_df[treatment_col].isin(valid_groups)].copy()

    chi2_stat, p_value = survdiff(
        km_df[time_col].to_numpy(),
        km_df[event_col].to_numpy(),
        km_df[treatment_col].to_numpy(),
    )

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for group in valid_groups:
        subset = km_df[km_df[treatment_col] == group]
        _plot_km(
            time_values=subset[time_col].to_numpy(),
            event_values=subset[event_col].to_numpy(),
            label=f'{group} (n={len(subset)})',
            ax=ax,
        )

    ax.set_title(f'{title}\nLog-rank p = {p_value:.4g}, chi2 = {chi2_stat:.3f}')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Survival probability')
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.show()

    print(f'[{title}] Log-rank test: chi2={chi2_stat:.4f}, p={p_value:.6g}')
    print(f'[{title}] Groups used (n >= {min_group_n}):')
    print(km_df[treatment_col].value_counts().to_string())

    return {
        'chi2': float(chi2_stat),
        'p_value': float(p_value),
        'n_groups': int(len(valid_groups)),
        'n_rows': int(len(km_df)),
    }


def _plot_legacy_grouped_km(df, time_col, event_col, group_col, title, min_group_n=10):
    if group_col is None:
        print(f'[{title}] Skipped: grouping column not found.')
        return None

    km_df = df[[time_col, event_col, group_col]].copy()
    km_df[time_col] = pd.to_numeric(km_df[time_col], errors='coerce')
    km_df[event_col] = _normalize_event_binary(km_df[event_col])
    km_df[group_col] = km_df[group_col].astype(str).str.strip()
    km_df = km_df.dropna(subset=[time_col, event_col, group_col])
    km_df = km_df[km_df[time_col] >= 0]

    invalid_labels = {'', 'na', 'nan', 'none', 'unknown'}
    km_df = km_df[~km_df[group_col].str.lower().isin(invalid_labels)]

    counts = km_df[group_col].value_counts()
    valid_groups = counts[counts >= min_group_n].index.tolist()

    if len(valid_groups) < 2:
        print(f'[{title}] Skipped: fewer than 2 groups with at least {min_group_n} samples.')
        if not counts.empty:
            print('Observed groups and counts:')
            print(counts.to_string())
        return None

    km_df = km_df[km_df[group_col].isin(valid_groups)].copy()

    chi2_stat, p_value = survdiff(
        km_df[time_col].to_numpy(),
        km_df[event_col].to_numpy(),
        km_df[group_col].to_numpy(),
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    for group in valid_groups:
        subset = km_df[km_df[group_col] == group]
        _plot_km(
            time_values=subset[time_col].to_numpy(),
            event_values=subset[event_col].to_numpy(),
            label=f'{group} (n={len(subset)})',
            ax=ax,
        )

    ax.set_title(f'{title}\nLog-rank p = {p_value:.4g}, chi2 = {chi2_stat:.3f}')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Survival probability')
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.show()

    print(f'[{title}] Log-rank test: chi2={chi2_stat:.4f}, p={p_value:.6g}')
    print(f'[{title}] Groups used (n >= {min_group_n}):')
    print(km_df[group_col].value_counts().to_string())

    return {
        'chi2': float(chi2_stat),
        'p_value': float(p_value),
        'n_groups': int(len(valid_groups)),
        'n_rows': int(len(km_df)),
    }


def run_km_analyses(merged_df, min_group_n=10):
    broad_treatment_col = _find_column(
        merged_df,
        ['Broad_treatment', 'broad_treatment', 'Broad Treatment', 'broad treatment'],
    )

    if broad_treatment_col is None:
        raise ValueError('Could not find Broad_treatment column required for KM subgroup plots.')

    endpoint_specs = [
        {
            'name': 'PFS',
            'time_col': _find_column(merged_df, ['PFS_days', 'pfs_days']),
            'event_col': _find_column(merged_df, ['PFS_censoring', 'pfs_censoring', 'PFS_event', 'pfs_event']),
            'event_text': '1 = progression event, 0 = censored',
        },
        {
            'name': 'OS',
            'time_col': _find_column(merged_df, ['OS_days', 'os_days', 'overall_survival_days']),
            'event_col': _find_column(merged_df, ['OS_censoring', 'os_censoring', 'OS_event', 'os_event', 'censoring']),
            'event_text': '1 = death event, 0 = censored',
        },
    ]

    km_results = {}
    for endpoint in endpoint_specs:
        endpoint_name = endpoint['name']
        time_col = endpoint['time_col']
        event_col = endpoint['event_col']

        print('')
        print('=' * 90)
        print(f'Endpoint: {endpoint_name}')
        print(f'  Time column: {time_col}')
        print(f'  Event column: {event_col} ({endpoint["event_text"]})')

        if time_col is None or event_col is None:
            print(f'[{endpoint_name}] Skipped: required time/event columns not found.')
            continue

        stats_out = _plot_grouped_km_by_treatment(
            df=merged_df,
            time_col=time_col,
            event_col=event_col,
            treatment_col=broad_treatment_col,
            title=f'Kaplan-Meier by Broad_treatment ({endpoint_name})',
            min_group_n=min_group_n,
        )
        km_results[endpoint_name] = stats_out

    print('')
    print('KM summary by Broad_treatment:')
    display(pd.DataFrame([{'endpoint': k, **(v if isinstance(v, dict) else {})} for k, v in km_results.items()]))

    print('')
    print('Legacy OS Kaplan-Meier curves:')
    os_time_col = _find_column(merged_df, ['OS_days', 'os_days', 'overall_survival_days'])
    os_event_col = _find_column(merged_df, ['OS_censoring', 'os_censoring', 'OS_event', 'os_event', 'censoring'])
    legacy_os_stats = {}

    if os_time_col is None or os_event_col is None:
        print('[Legacy OS KM] Skipped: required OS time/event columns not found.')
    else:
        histology_col = _find_column(merged_df, ['Histology', 'histology'])
        stage_col = _find_column(merged_df, ['Stage', 'stage', 'Clinical_stage', 'clinical_stage'])
        smoking_col = _find_column(merged_df, ['Smoking', 'smoking', 'smoking_status', 'Smoking_status'])

        legacy_os_stats['Histology'] = _plot_legacy_grouped_km(
            merged_df,
            os_time_col,
            os_event_col,
            histology_col,
            'Kaplan-Meier by Histology (OS)',
            min_group_n=min_group_n,
        )
        legacy_os_stats['Stage'] = _plot_legacy_grouped_km(
            merged_df,
            os_time_col,
            os_event_col,
            stage_col,
            'Kaplan-Meier by Stage (OS)',
            min_group_n=min_group_n,
        )
        legacy_os_stats['Smoking'] = _plot_legacy_grouped_km(
            merged_df,
            os_time_col,
            os_event_col,
            smoking_col,
            'Kaplan-Meier by Smoking (OS)',
            min_group_n=min_group_n,
        )

    print('')
    print('Legacy OS KM summary:')
    display(pd.DataFrame([
        {'grouping': key, **(value if isinstance(value, dict) else {})}
        for key, value in legacy_os_stats.items()
    ]))

    return km_results, legacy_os_stats


def _prepare_cox_design(df, time_col, event_col, corr_threshold=0.95):
    cluster_cols = [
        column
        for column in df.columns
        if column.startswith('freq_cluster_') or column.startswith('density_cluster_')
    ]

    categorical_candidates = [
        ['Histology', 'histology'],
        ['Stage', 'stage', 'Clinical_stage', 'clinical_stage'],
        ['Smoking', 'smoking', 'smoking_status', 'Smoking_status'],
        ['Tissue_origin', 'tissue_origin', 'Tissue Origin', 'tissue origin'],
        ['Sample_location', 'sample_location', 'Sample Location', 'sample location'],
        ['Sex', 'sex'],
        ['Age_dicho', 'age_dicho'],
    ]
    categorical_cols = [
        found
        for found in (_find_column(df, candidates) for candidates in categorical_candidates)
        if found is not None
    ]

    model_df = df[[time_col, event_col] + cluster_cols + categorical_cols].copy()
    model_df[time_col] = pd.to_numeric(model_df[time_col], errors='coerce')
    model_df[event_col] = _normalize_event_binary(model_df[event_col])

    for column in cluster_cols:
        model_df[column] = pd.to_numeric(model_df[column], errors='coerce')

    model_df = model_df.dropna(subset=[time_col, event_col])
    model_df = model_df[model_df[time_col] >= 0]

    if categorical_cols:
        model_df[categorical_cols] = model_df[categorical_cols].astype(str).apply(lambda col: col.str.strip())
        model_df[categorical_cols] = model_df[categorical_cols].replace({'': np.nan, 'NA': np.nan, 'nan': np.nan})

    base_cols = [time_col, event_col] + cluster_cols
    model_df = model_df[base_cols + categorical_cols]
    model_df = model_df.dropna()

    design_df = pd.get_dummies(
        model_df.drop(columns=[time_col, event_col]),
        columns=categorical_cols,
        drop_first=True,
        dtype=float,
    )

    if design_df.empty:
        return None, None

    variances = design_df.var(axis=0)
    keep_cols = variances[variances > 1e-8].index.tolist()
    design_df = design_df[keep_cols]

    if design_df.empty:
        return None, None
    if corr_threshold is not None and 0.0 < corr_threshold < 1.0 and design_df.shape[1] > 1:
        corr_matrix = design_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if (upper[col] > corr_threshold).any()]
        if to_drop:
            design_df = design_df.drop(columns=to_drop)
            print(
                f'[Cox design] Dropped {len(to_drop)} highly correlated features '
                f'(threshold={corr_threshold}).'
            )
            print('[Cox design] Removed features:')
            for feature_name in to_drop:
                print(f'  - {feature_name}')

    design_df = (design_df - design_df.mean(axis=0)) / design_df.std(axis=0)
    design_df = design_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    col_min = design_df.min(axis=0)
    col_max = design_df.max(axis=0)
    denom = (col_max - col_min).replace(0, np.nan)
    design_df = ((design_df - col_min) / denom).clip(lower=0.0, upper=1.0)
    design_df = design_df.dropna(axis=1)

    if design_df.empty:
        return None, None

    final = pd.concat([model_df[[time_col, event_col]].reset_index(drop=True), design_df.reset_index(drop=True)], axis=1)
    final = final.dropna()

    if final.empty:
        return None, None

    return final, list(design_df.columns)


def _summarize_cox_result(result_name, result_obj, feature_names, top_n=12):
    params = pd.Series(result_obj.params, index=feature_names, name='coef')
    params = params.replace([np.inf, -np.inf], np.nan).dropna()
    non_zero = int((params.abs() > 1e-8).sum())

    out = pd.DataFrame({'coef': params, 'hazard_ratio': np.exp(params)})
    out['abs_coef'] = out['coef'].abs()
    out = out.sort_values('abs_coef', ascending=False)

    print(f'[{result_name}] Features: {len(params)} | Non-zero coefficients: {non_zero}')
    display(out.head(top_n).drop(columns=['abs_coef']))
    return out


def _fit_cox_family(cox_df, time_col, event_col, feature_cols):
    endog_time = cox_df[time_col].to_numpy()
    status = cox_df[event_col].to_numpy()
    exog = cox_df[feature_cols].to_numpy()

    cox_model = PHReg(endog=endog_time, exog=exog, status=status)

    raw_results = {}
    coef_tables = {}

    try:
        res_lasso = cox_model.fit_regularized(method='elastic_net', alpha=0.05, L1_wt=1.0, maxiter=300)
        raw_results['lasso'] = res_lasso
        coef_tables['lasso'] = _summarize_cox_result('Lasso Cox (L1)', res_lasso, feature_cols)
    except Exception as exc:
        print(f'[Lasso Cox] Failed: {exc}')

    return raw_results, coef_tables


def run_cox_models_by_treatment(merged_df, min_subgroup_n=20, corr_threshold=0.95):
    broad_treatment_col = _find_column(
        merged_df,
        ['Broad_treatment', 'broad_treatment', 'Broad Treatment', 'broad treatment'],
    )
    if broad_treatment_col is None:
        raise ValueError('Could not find Broad_treatment column for subgroup analysis.')

    endpoint_specs = [
        {
            'name': 'PFS',
            'time_col': _find_column(merged_df, ['PFS_days', 'pfs_days']),
            'event_col': _find_column(merged_df, ['PFS_censoring', 'pfs_censoring', 'PFS_event', 'pfs_event']),
        },
        {
            'name': 'OS',
            'time_col': _find_column(merged_df, ['OS_days', 'os_days', 'overall_survival_days']),
            'event_col': _find_column(merged_df, ['OS_censoring', 'os_censoring', 'OS_event', 'os_event', 'censoring']),
        },
    ]

    clean_treatment = merged_df[broad_treatment_col].astype(str).str.strip()
    valid_mask = ~clean_treatment.str.lower().isin({'', 'na', 'nan', 'none', 'unknown'})
    treatment_counts = clean_treatment[valid_mask].value_counts()
    valid_treatments = treatment_counts[treatment_counts >= min_subgroup_n].index.tolist()

    if len(valid_treatments) == 0:
        raise ValueError(
            f'No Broad_treatment subgroup has at least {min_subgroup_n} samples. '
            'Lower min_subgroup_n if needed.'
        )

    print(f'Broad_treatment column: {broad_treatment_col}')
    print(f'Subgroups retained (n >= {min_subgroup_n}):')
    print(treatment_counts[treatment_counts >= min_subgroup_n].to_string())

    cox_run_registry = {}
    cox_overview_rows = []

    for endpoint in endpoint_specs:
        endpoint_name = endpoint['name']
        time_col = endpoint['time_col']
        event_col = endpoint['event_col']

        print('')
        print('=' * 90)
        print(f'Endpoint: {endpoint_name}')
        print(f'  Time column: {time_col}')
        print(f'  Event column: {event_col} (expected coding: 1=event, 0=censored)')

        if time_col is None or event_col is None:
            print(f'[{endpoint_name}] Skipped: required time/event columns not found.')
            continue

        for treatment_name in valid_treatments:
            subgroup_df = merged_df[clean_treatment == treatment_name].copy()
            print('')
            print('-' * 90)
            print(f'Subgroup: {treatment_name} | Raw rows: {len(subgroup_df)}')

            cox_df, feature_cols = _prepare_cox_design(
                subgroup_df,
                time_col=time_col,
                event_col=event_col,
                corr_threshold=corr_threshold,
            )
            if cox_df is None or feature_cols is None:
                print(f'[{endpoint_name} | {treatment_name}] Skipped: design matrix became empty after cleaning.')
                continue

            if len(cox_df) < 30:
                print(f'[{endpoint_name} | {treatment_name}] Skipped: too few complete rows ({len(cox_df)}).')
                continue

            if len(feature_cols) < 2:
                print(f'[{endpoint_name} | {treatment_name}] Skipped: too few covariates ({len(feature_cols)}).')
                continue

            print(f'[{endpoint_name} | {treatment_name}] Rows used: {len(cox_df)} | Covariates: {len(feature_cols)}')

            raw_results, coef_tables = _fit_cox_family(
                cox_df=cox_df,
                time_col=time_col,
                event_col=event_col,
                feature_cols=feature_cols,
            )

            if not raw_results:
                print(f'[{endpoint_name} | {treatment_name}] All model fits failed.')
                continue

            selection_summary = []
            for model_name, coef_df in coef_tables.items():
                selection_summary.append({
                    'endpoint': endpoint_name,
                    'broad_treatment': treatment_name,
                    'model': model_name,
                    'n_nonzero': int((coef_df['coef'].abs() > 1e-8).sum()),
                    'n_features': int(len(coef_df)),
                    'n_rows': int(len(cox_df)),
                })

            selection_summary_df = pd.DataFrame(selection_summary).sort_values('n_nonzero')
            print(f'Model sparsity summary [{endpoint_name} | {treatment_name}]:')
            display(selection_summary_df)

            cox_run_registry[(endpoint_name, treatment_name)] = {
                'time_col': time_col,
                'event_col': event_col,
                'feature_cols': feature_cols,
                'cox_df': cox_df,
                'raw_results': raw_results,
                'coef_tables': coef_tables,
                'selection_summary_df': selection_summary_df,
            }

            cox_overview_rows.extend(selection_summary)

    cox_overview_df = pd.DataFrame(cox_overview_rows)
    print('')
    print('Global Cox run overview:')
    if cox_overview_df.empty:
        print('No successful endpoint/subgroup Cox fits were produced.')
    else:
        display(cox_overview_df.sort_values(['endpoint', 'broad_treatment', 'n_nonzero']))

    return cox_run_registry, cox_overview_df


def clean_group_labels(series):
    cleaned = series.astype(str).str.strip()
    invalid = {'', 'na', 'nan', 'none', 'unknown', 'not available'}
    return cleaned.where(~cleaned.str.lower().isin(invalid))


def get_cluster_feature_columns(df):
    return [
        column
        for column in df.columns
        if column.startswith('freq_cluster_') or column.startswith('density_cluster_')
    ]


def compare_cluster_features_by_group(df, group_col, title, cluster_cols, min_group_n=5, top_n_features=6):
    if group_col is None:
        print(f'[{title}] Skipped: required column was not found.')
        return None

    if not cluster_cols:
        print(f'[{title}] Skipped: no cluster feature columns were found.')
        return None

    analysis_df = df[[group_col] + cluster_cols].copy()
    analysis_df[group_col] = clean_group_labels(analysis_df[group_col])

    for column in cluster_cols:
        analysis_df[column] = pd.to_numeric(analysis_df[column], errors='coerce')

    analysis_df = analysis_df.dropna(subset=[group_col])
    group_counts = analysis_df[group_col].value_counts()
    valid_groups = group_counts[group_counts >= min_group_n].index.tolist()

    if len(valid_groups) < 2:
        print(f'[{title}] Skipped: fewer than 2 groups with at least {min_group_n} samples.')
        if not group_counts.empty:
            print('Observed groups and counts:')
            print(group_counts.to_string())
        return None

    analysis_df = analysis_df[analysis_df[group_col].isin(valid_groups)]

    plottable_features = []
    for feature in cluster_cols:
        if analysis_df[feature].notna().sum() > 0:
            plottable_features.append(feature)

    if not plottable_features:
        print(f'[{title}] No plottable cluster features found.')
        return None

    print(f'[{title}] Group counts (n >= {min_group_n} kept):')
    print(analysis_df[group_col].value_counts().to_string())
    print('')
    print(f'[{title}] Plotting all cluster features ({len(plottable_features)} total).')

    n_features = len(plottable_features)
    n_cols = min(3, n_features)
    n_rows = math.ceil(n_features / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)

    for index, feature in enumerate(plottable_features):
        row_idx = index // n_cols
        col_idx = index % n_cols
        ax = axes[row_idx][col_idx]

        sns.boxplot(data=analysis_df, x=group_col, y=feature, ax=ax)
        sns.stripplot(
            data=analysis_df,
            x=group_col,
            y=feature,
            color='black',
            alpha=0.25,
            size=2.5,
            jitter=0.2,
            ax=ax,
        )

        ax.set_title(feature)
        ax.set_xlabel(group_col)
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=30)

    for index in range(n_features, n_rows * n_cols):
        row_idx = index // n_cols
        col_idx = index % n_cols
        axes[row_idx][col_idx].axis('off')

    fig.suptitle(f'{title}: Top Cluster Feature Differences', y=1.02)
    plt.tight_layout()
    plt.show()

    return {
        'group_counts': analysis_df[group_col].value_counts(),
        'plotted_features': plottable_features,
        'n_features': len(plottable_features),
    }


def harrell_c_index(time, event, risk_score):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk_score = np.asarray(risk_score, dtype=float)

    concordant = 0.0
    comparable = 0.0
    ties = 0.0

    n = len(time)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = time[i], time[j]
            ei, ej = event[i], event[j]
            ri, rj = risk_score[i], risk_score[j]

            if ti == tj:
                continue

            if ti < tj and ei == 1:
                comparable += 1
                if ri > rj:
                    concordant += 1
                elif ri == rj:
                    ties += 1
            elif tj < ti and ej == 1:
                comparable += 1
                if rj > ri:
                    concordant += 1
                elif ri == rj:
                    ties += 1

    if comparable == 0:
        return np.nan
    return (concordant + 0.5 * ties) / comparable


def _safe_loglike(result_obj):
    llf = getattr(result_obj, 'llf', None)
    if llf is not None:
        return float(llf)

    model = getattr(result_obj, 'model', None)
    params = getattr(result_obj, 'params', None)
    if model is not None and params is not None:
        try:
            return float(model.loglike(params))
        except Exception:
            return np.nan
    return np.nan


def _risk_group_km_plot(df, time_col, event_col, risk_col, title):
    local = df[[time_col, event_col, risk_col]].dropna().copy()
    if local.empty:
        print(f'[{title}] Skipped: no data for risk-stratified KM.')
        return np.nan

    # qcut may fail when many risk scores are tied and 3 quantile bins cannot be formed.
    # In that case, fall back to fewer bins with matching labels.
    base_labels = ['Low risk', 'Mid risk', 'High risk']
    local['risk_group'] = np.nan
    for n_bins in (3, 2):
        try:
            labels = base_labels[:n_bins]
            local['risk_group'] = pd.qcut(
                local[risk_col],
                q=n_bins,
                labels=labels,
                duplicates='drop',
            )
            break
        except ValueError:
            continue

    local = local.dropna(subset=['risk_group'])

    if local['risk_group'].nunique() < 2:
        print(f'[{title}] Skipped: less than 2 risk groups available.')
        return np.nan

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for grp in base_labels:
        subset = local[local['risk_group'] == grp]
        if subset.empty:
            continue
        sf = SurvfuncRight(subset[time_col].to_numpy(), subset[event_col].to_numpy())
        ax.step(sf.surv_times, sf.surv_prob, where='post', label=f'{grp} (n={len(subset)})')

    chi2_stat, p_value = survdiff(
        local[time_col].to_numpy(),
        local[event_col].to_numpy(),
        local['risk_group'].astype(str).to_numpy(),
    )

    ax.set_title(f'{title}\nRisk tertile log-rank p = {p_value:.4g}')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Survival probability')
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.show()

    return float(p_value)


def run_cox_diagnostics(cox_run_registry, max_plots=8):
    if not cox_run_registry:
        print('No successful Cox runs available for diagnostics.')
        return pd.DataFrame(), pd.DataFrame()

    diagnostics_rows = []
    risk_plot_candidates = []

    for (endpoint_name, treatment_name), run_data in cox_run_registry.items():
        cox_df = run_data['cox_df']
        time_col = run_data['time_col']
        event_col = run_data['event_col']
        feature_cols = run_data['feature_cols']
        raw_results = run_data['raw_results']

        for model_name, result_obj in raw_results.items():
            params = np.asarray(result_obj.params, dtype=float)
            risk = np.asarray(cox_df[feature_cols].to_numpy() @ params, dtype=float)
            cidx = harrell_c_index(
                time=cox_df[time_col].to_numpy(),
                event=cox_df[event_col].to_numpy(),
                risk_score=risk,
            )
            llf = _safe_loglike(result_obj)
            n_nonzero = int(np.sum(np.abs(params) > 1e-8))

            diagnostics_rows.append({
                'endpoint': endpoint_name,
                'broad_treatment': treatment_name,
                'model': model_name,
                'n_rows': int(len(cox_df)),
                'n_features': int(len(feature_cols)),
                'n_nonzero': n_nonzero,
                'c_index': cidx,
                'log_likelihood': llf,
            })

            risk_plot_candidates.append({
                'endpoint': endpoint_name,
                'broad_treatment': treatment_name,
                'model': model_name,
                'cox_df': cox_df,
                'time_col': time_col,
                'event_col': event_col,
                'risk': risk,
                'c_index': cidx,
            })

    diagnostics_df = pd.DataFrame(diagnostics_rows).sort_values(
        ['endpoint', 'broad_treatment', 'c_index'],
        ascending=[True, True, False],
    )

    print('Cox diagnostics summary by endpoint and Broad_treatment:')
    display(diagnostics_df)

    best_models = diagnostics_df.groupby(['endpoint', 'broad_treatment'], as_index=False).first()
    print('Best model per endpoint/subgroup (by C-index):')
    display(best_models[['endpoint', 'broad_treatment', 'model', 'c_index', 'n_nonzero', 'n_rows']])

    best_lookup = {
        (row['endpoint'], row['broad_treatment']): row['model']
        for _, row in best_models.iterrows()
    }

    plotted = 0
    for candidate in risk_plot_candidates:
        key = (candidate['endpoint'], candidate['broad_treatment'])
        if best_lookup.get(key) != candidate['model']:
            continue

        if plotted >= max_plots:
            print(f'Skipping additional risk KM plots after {max_plots} panels to keep output compact.')
            break

        local_frame = candidate['cox_df'][[candidate['time_col'], candidate['event_col']]].copy()
        local_frame['risk_score'] = candidate['risk']

        title = (
            f"Risk-stratified KM | {candidate['endpoint']} | {candidate['broad_treatment']} | "
            f"{candidate['model']}"
        )
        p_value = _risk_group_km_plot(
            df=local_frame,
            time_col=candidate['time_col'],
            event_col=candidate['event_col'],
            risk_col='risk_score',
            title=title,
        )
        print(
            f"[{candidate['endpoint']} | {candidate['broad_treatment']} | {candidate['model']}] "
            f"risk-group log-rank p = {p_value:.6g}"
        )
        plotted += 1

    return diagnostics_df, best_models


def run_univariate_cox_for_lasso_selected(cox_run_registry, coef_threshold=1e-8):
    if not cox_run_registry:
        print('No successful Cox runs available for univariate follow-up.')
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    skipped = []

    for (endpoint_name, treatment_name), run_data in cox_run_registry.items():
        cox_df = run_data['cox_df']
        time_col = run_data['time_col']
        event_col = run_data['event_col']
        coef_tables = run_data.get('coef_tables', {})

        lasso_table = coef_tables.get('lasso')
        if lasso_table is None or lasso_table.empty:
            skipped.append({
                'endpoint': endpoint_name,
                'broad_treatment': treatment_name,
                'reason': 'No lasso coefficient table available.',
            })
            continue

        selected = lasso_table[lasso_table['coef'].abs() > coef_threshold].copy()
        if selected.empty:
            skipped.append({
                'endpoint': endpoint_name,
                'broad_treatment': treatment_name,
                'reason': 'No non-zero lasso coefficients.',
            })
            continue

        for feature_name, feature_row in selected.iterrows():
            local = cox_df[[time_col, event_col, feature_name]].dropna().copy()

            if local.empty or local[feature_name].nunique(dropna=True) < 2:
                rows.append({
                    'endpoint': endpoint_name,
                    'broad_treatment': treatment_name,
                    'feature': feature_name,
                    'lasso_coef': float(feature_row['coef']),
                    'n_rows': int(len(local)),
                    'uni_coef': np.nan,
                    'uni_hazard_ratio': np.nan,
                    'uni_ci95_low': np.nan,
                    'uni_ci95_high': np.nan,
                    'uni_p_value': np.nan,
                    'uni_log_likelihood': np.nan,
                    'status': 'skipped_low_variance_or_empty',
                })
                continue

            try:
                model = PHReg(
                    endog=local[time_col].to_numpy(),
                    exog=local[[feature_name]].to_numpy(),
                    status=local[event_col].to_numpy(),
                )
                result = model.fit(disp=0)

                beta = float(np.asarray(result.params, dtype=float)[0])
                bse_arr = getattr(result, 'bse', None)
                if bse_arr is not None:
                    se = float(np.asarray(bse_arr, dtype=float)[0])
                    ci_low = float(np.exp(beta - 1.96 * se))
                    ci_high = float(np.exp(beta + 1.96 * se))
                else:
                    ci_low = np.nan
                    ci_high = np.nan

                pvalues_arr = getattr(result, 'pvalues', None)
                if pvalues_arr is not None:
                    p_value = float(np.asarray(pvalues_arr, dtype=float)[0])
                else:
                    p_value = np.nan

                rows.append({
                    'endpoint': endpoint_name,
                    'broad_treatment': treatment_name,
                    'feature': feature_name,
                    'lasso_coef': float(feature_row['coef']),
                    'n_rows': int(len(local)),
                    'uni_coef': beta,
                    'uni_hazard_ratio': float(np.exp(beta)),
                    'uni_ci95_low': ci_low,
                    'uni_ci95_high': ci_high,
                    'uni_p_value': p_value,
                    'uni_log_likelihood': _safe_loglike(result),
                    'status': 'ok',
                })
            except Exception as exc:
                rows.append({
                    'endpoint': endpoint_name,
                    'broad_treatment': treatment_name,
                    'feature': feature_name,
                    'lasso_coef': float(feature_row['coef']),
                    'n_rows': int(len(local)),
                    'uni_coef': np.nan,
                    'uni_hazard_ratio': np.nan,
                    'uni_ci95_low': np.nan,
                    'uni_ci95_high': np.nan,
                    'uni_p_value': np.nan,
                    'uni_log_likelihood': np.nan,
                    'status': f'failed: {exc}',
                })

    univariate_df = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)

    print('Univariate Cox follow-up for lasso-selected features:')
    if univariate_df.empty:
        print('No univariate models were fit.')
    else:
        univariate_df = univariate_df.sort_values(
            ['endpoint', 'broad_treatment', 'uni_p_value', 'feature'],
            na_position='last',
        )
        display(univariate_df)

        sig = univariate_df[(univariate_df['status'] == 'ok') & (univariate_df['uni_p_value'] < 0.05)]
        print('Significant univariate associations (p < 0.05):')
        if sig.empty:
            print('None found at p < 0.05.')
        else:
            display(sig[['endpoint', 'broad_treatment', 'feature', 'lasso_coef', 'uni_hazard_ratio', 'uni_p_value']])

    if not skipped_df.empty:
        print('Subgroups skipped before univariate fitting:')
        display(skipped_df)

    return univariate_df, skipped_df


def run_all_univariate_cox(cox_run_registry):
    """Run univariate Cox PH for every feature in each endpoint/treatment subgroup."""
    if not cox_run_registry:
        print('No successful Cox runs available for univariate analysis.')
        return pd.DataFrame()

    rows = []

    for (endpoint_name, treatment_name), run_data in cox_run_registry.items():
        cox_df = run_data['cox_df']
        time_col = run_data['time_col']
        event_col = run_data['event_col']
        feature_cols = run_data['feature_cols']

        print(f'[{endpoint_name} | {treatment_name}] Running univariate Cox for {len(feature_cols)} features ...')

        for feature_name in feature_cols:
            local = cox_df[[time_col, event_col, feature_name]].dropna().copy()

            if local.empty or local[feature_name].nunique(dropna=True) < 2:
                rows.append({
                    'endpoint': endpoint_name,
                    'broad_treatment': treatment_name,
                    'feature': feature_name,
                    'n_rows': int(len(local)),
                    'uni_coef': np.nan,
                    'uni_hazard_ratio': np.nan,
                    'uni_ci95_low': np.nan,
                    'uni_ci95_high': np.nan,
                    'uni_p_value': np.nan,
                    'status': 'skipped_low_variance_or_empty',
                })
                continue

            try:
                model = PHReg(
                    endog=local[time_col].to_numpy(),
                    exog=local[[feature_name]].to_numpy(),
                    status=local[event_col].to_numpy(),
                )
                result = model.fit(disp=0)

                beta = float(np.asarray(result.params, dtype=float)[0])
                bse_arr = getattr(result, 'bse', None)
                if bse_arr is not None:
                    se = float(np.asarray(bse_arr, dtype=float)[0])
                    ci_low = float(np.exp(beta - 1.96 * se))
                    ci_high = float(np.exp(beta + 1.96 * se))
                else:
                    ci_low = np.nan
                    ci_high = np.nan

                pvalues_arr = getattr(result, 'pvalues', None)
                if pvalues_arr is not None:
                    p_value = float(np.asarray(pvalues_arr, dtype=float)[0])
                else:
                    p_value = np.nan

                rows.append({
                    'endpoint': endpoint_name,
                    'broad_treatment': treatment_name,
                    'feature': feature_name,
                    'n_rows': int(len(local)),
                    'uni_coef': beta,
                    'uni_hazard_ratio': float(np.exp(beta)),
                    'uni_ci95_low': ci_low,
                    'uni_ci95_high': ci_high,
                    'uni_p_value': p_value,
                    'status': 'ok',
                })
            except Exception as exc:
                rows.append({
                    'endpoint': endpoint_name,
                    'broad_treatment': treatment_name,
                    'feature': feature_name,
                    'n_rows': int(len(local)),
                    'uni_coef': np.nan,
                    'uni_hazard_ratio': np.nan,
                    'uni_ci95_low': np.nan,
                    'uni_ci95_high': np.nan,
                    'uni_p_value': np.nan,
                    'status': f'failed: {exc}',
                })

    all_uni_df = pd.DataFrame(rows)
    if not all_uni_df.empty:
        all_uni_df = all_uni_df.sort_values(
            ['endpoint', 'broad_treatment', 'uni_p_value'],
            na_position='last',
        )

    ok_count = int((all_uni_df['status'] == 'ok').sum()) if not all_uni_df.empty else 0
    sig_count = int(((all_uni_df['status'] == 'ok') & (all_uni_df['uni_p_value'] < 0.05)).sum()) if not all_uni_df.empty else 0

    print(f'\nAll univariate Cox results: {len(all_uni_df)} feature-endpoint-treatment combinations.')
    print(f'  Successfully fit: {ok_count}')
    print(f'  Significant at p < 0.05: {sig_count}')
    display(all_uni_df)

    return all_uni_df


def compare_univariate_vs_lasso(all_univariate_df, cox_run_registry, coef_threshold=1e-8, p_threshold=0.05):
    """Compare univariate significance against Lasso-selected features."""
    if all_univariate_df.empty or not cox_run_registry:
        print('No data available for univariate vs. lasso comparison.')
        return pd.DataFrame()

    comparison_rows = []

    for (endpoint_name, treatment_name), run_data in cox_run_registry.items():
        coef_tables = run_data.get('coef_tables', {})
        lasso_table = coef_tables.get('lasso')
        if lasso_table is None or lasso_table.empty:
            continue

        lasso_selected = set(lasso_table[lasso_table['coef'].abs() > coef_threshold].index)

        uni_subset = all_univariate_df[
            (all_univariate_df['endpoint'] == endpoint_name)
            & (all_univariate_df['broad_treatment'] == treatment_name)
            & (all_univariate_df['status'] == 'ok')
        ]

        for _, row in uni_subset.iterrows():
            feature = row['feature']
            is_lasso = feature in lasso_selected
            is_uni_sig = row['uni_p_value'] < p_threshold
            lasso_coef = float(lasso_table.loc[feature, 'coef']) if feature in lasso_table.index else 0.0

            if is_lasso and is_uni_sig:
                agreement = 'both'
            elif is_lasso:
                agreement = 'lasso_only'
            elif is_uni_sig:
                agreement = 'uni_only'
            else:
                agreement = 'neither'

            comparison_rows.append({
                'endpoint': endpoint_name,
                'broad_treatment': treatment_name,
                'feature': feature,
                'lasso_selected': is_lasso,
                'lasso_coef': lasso_coef,
                'uni_significant': is_uni_sig,
                'uni_hazard_ratio': row['uni_hazard_ratio'],
                'uni_p_value': row['uni_p_value'],
                'uni_ci95_low': row['uni_ci95_low'],
                'uni_ci95_high': row['uni_ci95_high'],
                'agreement': agreement,
            })

    comparison_df = pd.DataFrame(comparison_rows)
    if comparison_df.empty:
        print('No comparison data produced.')
        return comparison_df

    comparison_df = comparison_df.sort_values(
        ['endpoint', 'broad_treatment', 'agreement', 'uni_p_value'],
        na_position='last',
    )

    for (endpoint, treatment), group in comparison_df.groupby(['endpoint', 'broad_treatment']):
        n_both = int((group['agreement'] == 'both').sum())
        n_lasso_only = int((group['agreement'] == 'lasso_only').sum())
        n_uni_only = int((group['agreement'] == 'uni_only').sum())
        n_neither = int((group['agreement'] == 'neither').sum())
        print(f'\n[{endpoint} | {treatment}] Univariate vs Lasso comparison:')
        print(f'  Both selected & significant: {n_both}')
        print(f'  Lasso-selected only:         {n_lasso_only}')
        print(f'  Univariate-significant only: {n_uni_only}')
        print(f'  Neither:                     {n_neither}')

        agreed = group[group['agreement'] == 'both']
        if not agreed.empty:
            print('  Concordant features:')
            display(agreed[['feature', 'lasso_coef', 'uni_hazard_ratio', 'uni_p_value']])

    display(comparison_df)
    return comparison_df


def plot_forest_by_treatment(all_univariate_df, cox_run_registry, coef_threshold=1e-8, p_threshold=0.05, max_features=30):
    """Generate forest plots of univariate HR per endpoint/treatment, highlighting Lasso-selected features."""
    if all_univariate_df.empty:
        print('No univariate results available for forest plots.')
        return

    for (endpoint_name, treatment_name), run_data in cox_run_registry.items():
        coef_tables = run_data.get('coef_tables', {})
        lasso_table = coef_tables.get('lasso')
        lasso_selected = set()
        if lasso_table is not None and not lasso_table.empty:
            lasso_selected = set(lasso_table[lasso_table['coef'].abs() > coef_threshold].index)

        subset = all_univariate_df[
            (all_univariate_df['endpoint'] == endpoint_name)
            & (all_univariate_df['broad_treatment'] == treatment_name)
            & (all_univariate_df['status'] == 'ok')
        ].copy()

        if subset.empty:
            print(f'[{endpoint_name} | {treatment_name}] No valid univariate results for forest plot.')
            continue

        subset = subset.sort_values('uni_p_value').head(max_features).copy()
        subset = subset.iloc[::-1].reset_index(drop=True)

        subset['is_lasso'] = subset['feature'].isin(lasso_selected)
        subset['is_significant'] = subset['uni_p_value'] < p_threshold

        fig_height = max(4, 0.35 * len(subset) + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        for i, (_, row) in enumerate(subset.iterrows()):
            hr = row['uni_hazard_ratio']
            ci_low = row['uni_ci95_low']
            ci_high = row['uni_ci95_high']

            if row['is_lasso'] and row['is_significant']:
                color = '#d62728'
                marker = 'D'
            elif row['is_lasso']:
                color = '#ff7f0e'
                marker = 's'
            elif row['is_significant']:
                color = '#1f77b4'
                marker = 'o'
            else:
                color = '#7f7f7f'
                marker = 'o'

            if pd.notna(ci_low) and pd.notna(ci_high):
                ax.plot([ci_low, ci_high], [i, i], color=color, linewidth=1.5, solid_capstyle='round')
            ax.plot(hr, i, marker=marker, color=color, markersize=7, zorder=5)

        ax.axvline(x=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)

        y_positions = np.arange(len(subset))
        ax.set_yticks(y_positions)
        labels = []
        for _, row in subset.iterrows():
            p_str = f'p={row["uni_p_value"]:.3g}' if pd.notna(row['uni_p_value']) else 'p=NA'
            labels.append(f'{row["feature"]}  ({p_str})')
        ax.set_yticklabels(labels, fontsize=8)

        ax.set_xscale('log')
        ax.set_xlabel('Hazard Ratio (log scale)')
        ax.set_title(
            f'Forest Plot: {endpoint_name} | {treatment_name}\n'
            f'(top {len(subset)} features by univariate p-value)'
        )
        ax.grid(axis='x', alpha=0.2)

        legend_elements = [
            Line2D([0], [0], marker='D', color='#d62728', linestyle='None', markersize=7, label='Lasso + Uni sig'),
            Line2D([0], [0], marker='s', color='#ff7f0e', linestyle='None', markersize=7, label='Lasso only'),
            Line2D([0], [0], marker='o', color='#1f77b4', linestyle='None', markersize=7, label='Uni sig only'),
            Line2D([0], [0], marker='o', color='#7f7f7f', linestyle='None', markersize=7, label='Not selected'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8, frameon=True)

        plt.tight_layout()
        plt.show()

        n_lasso_shown = int(subset['is_lasso'].sum())
        n_sig_shown = int(subset['is_significant'].sum())
        print(f'[{endpoint_name} | {treatment_name}] Forest plot: {len(subset)} features shown.')
        print(f'  Lasso-selected in plot: {n_lasso_shown} | Univariate significant: {n_sig_shown}')


def run_univariate_analysis_and_forest_plots(cox_run_registry, coef_threshold=1e-8, p_threshold=0.05, max_features=30):
    """Run all-feature univariate Cox, compare vs. Lasso, and generate forest plots."""
    all_uni_df = run_all_univariate_cox(cox_run_registry)

    comparison_df = compare_univariate_vs_lasso(
        all_uni_df,
        cox_run_registry,
        coef_threshold=coef_threshold,
        p_threshold=p_threshold,
    )

    plot_forest_by_treatment(
        all_uni_df,
        cox_run_registry,
        coef_threshold=coef_threshold,
        p_threshold=p_threshold,
        max_features=max_features,
    )

    return all_uni_df, comparison_df
