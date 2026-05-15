import math
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff
from statsmodels.stats.multitest import multipletests

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


# ─── Utilities ──────────────────────────────────────────────────────────────────

_INVALID_LABELS = {'', 'na', 'nan', 'none', 'unknown', 'not available'}


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


def clean_group_labels(series):
    cleaned = series.astype(str).str.strip()
    return cleaned.where(~cleaned.str.lower().isin(_INVALID_LABELS))


def _cluster_sort_key(name):
    match = re.search(r'(\d+)', str(name))
    return int(match.group(1)) if match else 10_000


def get_cluster_feature_columns(df):
    cols = [c for c in df.columns if c.startswith('freq_cluster_')]
    return sorted(cols, key=_cluster_sort_key)


def exclude_background_clusters(df, background_cluster_ids, renormalize=True, verbose=True):
    """Drop `freq_cluster_{id}` columns for the listed cluster IDs.

    The cluster IDs follow the pickle scheme used everywhere downstream
    (0-indexed; matches `freq_cluster_*` column suffixes). Note that the
    cluster_image.jpg legend labels the same clusters 1..N — see the
    pipeline notes — so subtract 1 if you read IDs off the JPG legend.

    Parameters
    ----------
    df : pd.DataFrame
        Per-patient table with `freq_cluster_*` columns (typically `merged_df`).
    background_cluster_ids : iterable of int
        Cluster IDs to drop. Empty -> no-op (returns a copy).
    renormalize : bool, default True
        If True, rescale the remaining `freq_cluster_*` columns so each
        row sums to 1 (fraction of *non-background* tissue). If False,
        leave them as fraction of total tissue.
    verbose : bool, default True
        Print a one-line summary of what was dropped.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with the requested columns dropped (and optionally
        renormalized). Background IDs that aren't present in `df` are
        silently skipped.
    """
    out = df.copy()
    requested = [f'freq_cluster_{int(cid)}' for cid in background_cluster_ids]
    present = [c for c in requested if c in out.columns]
    missing = [c for c in requested if c not in out.columns]

    if not present:
        if verbose:
            print(f'[exclude_background] No matching freq_cluster_* columns found '
                  f'(asked for {len(requested)}); df returned unchanged.')
        return out

    out = out.drop(columns=present)

    if renormalize:
        remaining = get_cluster_feature_columns(out)
        if remaining:
            row_sums = out[remaining].sum(axis=1).replace(0, np.nan)
            out[remaining] = out[remaining].div(row_sums, axis=0).fillna(0.0)

    if verbose:
        print(f'[exclude_background] Dropped {len(present)} cluster column(s): {present}')
        if missing:
            print(f'  Skipped (not in df): {missing}')
        if renormalize:
            print('  Remaining freq_cluster_* columns renormalized to sum to 1 per row.')
        else:
            print('  Frequencies kept as fraction of total tissue (no renormalization).')

    return out


def _select_p_col(df, use_fdr):
    return 'p_fdr' if use_fdr and 'p_fdr' in df.columns else 'p'


def _ensure_parent_dir(path):
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)


_UNIVARIATE_NAN_FIELDS = {
    'uni_coef': np.nan,
    'uni_hazard_ratio': np.nan,
    'uni_ci95_low': np.nan,
    'uni_ci95_high': np.nan,
    'uni_p_value': np.nan,
    'uni_log_likelihood': np.nan,
}


def _univariate_skip_result(n_rows, status):
    return {'n_rows': int(n_rows), **_UNIVARIATE_NAN_FIELDS, 'status': status}


def _get_endpoint_specs(df):
    return [
        {
            'name': 'PFS',
            'time_col': _find_column(df, ['PFS_days', 'pfs_days']),
            'event_col': _find_column(df, ['PFS_censoring', 'pfs_censoring', 'PFS_event', 'pfs_event']),
            'event_text': '1 = progression event, 0 = censored',
        },
        {
            'name': 'OS',
            'time_col': _find_column(df, ['OS_days', 'os_days', 'overall_survival_days']),
            'event_col': _find_column(df, ['OS_censoring', 'os_censoring', 'OS_event', 'os_event', 'censoring']),
            'event_text': '1 = death event, 0 = censored',
        },
    ]


# ─── Cluster summarization ──────────────────────────────────────────────────────


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

        records.append(row)

    df = pd.DataFrame(records).fillna(0.0)
    meta_cols = ['patient_id', 'tissue_area_um2', 'n_tissue_superpixels']
    freq_cols = get_cluster_feature_columns(df)
    df = df[meta_cols + freq_cols]

    _ensure_parent_dir(output_path)
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


# ─── Patient matching & clinical merge ──────────────────────────────────────────


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
    match_records = [
        _best_match(cluster_id, clinical_ids, min_score=min_score)
        for cluster_id in cluster_df['cluster_patient_id']
    ]
    cluster_df[['matched_clinical_patient_id', 'match_score']] = pd.DataFrame(
        match_records, index=cluster_df.index, columns=['matched_clinical_patient_id', 'match_score']
    )
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

    priority_cols = [
        'patient_id',
        'cluster_patient_id',
        'clinical_patient_id',
        'matched_clinical_patient_id',
        'match_score',
        '_merge',
    ]
    priority_set = set(priority_cols)
    merged_df = merged_df[priority_cols + [c for c in merged_df.columns if c not in priority_set]]

    _ensure_parent_dir(output_path)
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


# ─── Kaplan-Meier ───────────────────────────────────────────────────────────────


def _plot_km(time_values, event_values, label, ax):
    sf = SurvfuncRight(time_values, event_values)
    ax.step(sf.surv_times, sf.surv_prob, where='post', label=label)


def _plot_grouped_km(df, time_col, event_col, group_col, title, min_group_n=10, figsize=(10, 6.5)):
    if group_col is None:
        print(f'[{title}] Skipped: grouping column not found.')
        return None

    km_df = df[[time_col, event_col, group_col]].copy()
    km_df[time_col] = pd.to_numeric(km_df[time_col], errors='coerce')
    km_df[event_col] = _normalize_event_binary(km_df[event_col])
    km_df[group_col] = clean_group_labels(km_df[group_col])
    km_df = km_df.dropna(subset=[time_col, event_col, group_col])
    km_df = km_df[km_df[time_col] >= 0]

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

    fig, ax = plt.subplots(figsize=figsize)
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

    endpoint_specs = _get_endpoint_specs(merged_df)

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

        stats_out = _plot_grouped_km(
            df=merged_df,
            time_col=time_col,
            event_col=event_col,
            group_col=broad_treatment_col,
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

        legacy_os_stats['Histology'] = _plot_grouped_km(
            merged_df, os_time_col, os_event_col, histology_col,
            'Kaplan-Meier by Histology (OS)', min_group_n=min_group_n, figsize=(9, 6),
        )
        legacy_os_stats['Stage'] = _plot_grouped_km(
            merged_df, os_time_col, os_event_col, stage_col,
            'Kaplan-Meier by Stage (OS)', min_group_n=min_group_n, figsize=(9, 6),
        )
        legacy_os_stats['Smoking'] = _plot_grouped_km(
            merged_df, os_time_col, os_event_col, smoking_col,
            'Kaplan-Meier by Smoking (OS)', min_group_n=min_group_n, figsize=(9, 6),
        )

    print('')
    print('Legacy OS KM summary:')
    display(pd.DataFrame([
        {'grouping': key, **(value if isinstance(value, dict) else {})}
        for key, value in legacy_os_stats.items()
    ]))

    return km_results, legacy_os_stats


# ─── Cox design & fitting ──────────────────────────────────────────────────────


def _prepare_cox_design(df, time_col, event_col, corr_threshold=0.95):
    cluster_cols = get_cluster_feature_columns(df)

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
            print(
                f'[Cox design] Dropped {len(to_drop)} highly correlated features '
                f'(threshold={corr_threshold}).'
            )
            print('[Cox design] Correlated pairs removed:')
            for feature_name in to_drop:
                correlated_with = upper.index[upper[feature_name] > corr_threshold].tolist()
                for partner in correlated_with:
                    corr_val = corr_matrix.loc[partner, feature_name]
                    print(f'  - {feature_name}  <->  {partner}  (r={corr_val:.4f})')
            design_df = design_df.drop(columns=to_drop)

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

    endpoint_specs = _get_endpoint_specs(merged_df)

    clean_treatment = clean_group_labels(merged_df[broad_treatment_col])
    treatment_counts = clean_treatment.dropna().value_counts()
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


# ─── Clinical comparisons ──────────────────────────────────────────────────────


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

    plottable_features = [f for f in cluster_cols if analysis_df[f].notna().any()]

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


# ─── Cox diagnostics ──────────────────────────────────────────────────────────


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


# ─── Univariate Cox ──────────────────────────────────────────────────────────


def _fit_univariate_cox(cox_df, time_col, event_col, feature_name):
    """Fit a single-feature Cox PH model and return a result dict."""
    local = cox_df[[time_col, event_col, feature_name]].dropna().copy()

    if local.empty or local[feature_name].nunique(dropna=True) < 2:
        return _univariate_skip_result(len(local), 'skipped_low_variance_or_empty')

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
        p_value = float(np.asarray(pvalues_arr, dtype=float)[0]) if pvalues_arr is not None else np.nan

        return {
            'n_rows': int(len(local)),
            'uni_coef': beta,
            'uni_hazard_ratio': float(np.exp(beta)),
            'uni_ci95_low': ci_low,
            'uni_ci95_high': ci_high,
            'uni_p_value': p_value,
            'uni_log_likelihood': _safe_loglike(result),
            'status': 'ok',
        }
    except Exception as exc:
        return _univariate_skip_result(len(local), f'failed: {exc}')


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
            uni_result = _fit_univariate_cox(cox_df, time_col, event_col, feature_name)
            rows.append({
                'endpoint': endpoint_name,
                'broad_treatment': treatment_name,
                'feature': feature_name,
                'lasso_coef': float(feature_row['coef']),
                **uni_result,
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
            uni_result = _fit_univariate_cox(cox_df, time_col, event_col, feature_name)
            rows.append({
                'endpoint': endpoint_name,
                'broad_treatment': treatment_name,
                'feature': feature_name,
                **uni_result,
            })

    all_uni_df = pd.DataFrame(rows)
    if all_uni_df.empty:
        ok_count = sig_count = 0
    else:
        all_uni_df = all_uni_df.sort_values(
            ['endpoint', 'broad_treatment', 'uni_p_value'],
            na_position='last',
        )
        ok_mask = all_uni_df['status'] == 'ok'
        ok_count = int(ok_mask.sum())
        sig_count = int((ok_mask & (all_uni_df['uni_p_value'] < 0.05)).sum())

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

        forest_style = {
            (True, True):   ('#d62728', 'D'),
            (True, False):  ('#ff7f0e', 's'),
            (False, True):  ('#1f77b4', 'o'),
            (False, False): ('#7f7f7f', 'o'),
        }
        for i, (_, row) in enumerate(subset.iterrows()):
            hr = row['uni_hazard_ratio']
            ci_low = row['uni_ci95_low']
            ci_high = row['uni_ci95_high']
            color, marker = forest_style[(bool(row['is_lasso']), bool(row['is_significant']))]

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


def plot_volcano_by_treatment(
    all_univariate_df,
    cox_run_registry,
    coef_threshold=1e-8,
    p_threshold=0.05,
    label_top_n=10,
    figsize=(9, 7),
):
    """Volcano plot of univariate Cox results per endpoint/treatment.

    X-axis: log2(HR)  (protective < 0 < risky)
    Y-axis: -log10(p-value)
    Lasso-selected features are drawn with a black edge ring.
    """
    if all_univariate_df.empty:
        print('No univariate results available for volcano plots.')
        return

    neg_log10_threshold = -math.log10(p_threshold)

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

        subset = subset[subset['uni_hazard_ratio'].notna() & subset['uni_p_value'].notna()]
        subset = subset[(subset['uni_hazard_ratio'] > 0) & np.isfinite(subset['uni_hazard_ratio'])]

        if subset.empty:
            print(f'[{endpoint_name} | {treatment_name}] No valid univariate results for volcano plot.')
            continue

        min_positive_p = subset.loc[subset['uni_p_value'] > 0, 'uni_p_value'].min()
        if pd.isna(min_positive_p):
            min_positive_p = 1e-300
        floor_p = max(min_positive_p * 0.1, 1e-300)
        p_plot = subset['uni_p_value'].clip(lower=floor_p)

        subset['log2_hr'] = np.log2(subset['uni_hazard_ratio'].astype(float))
        subset['neg_log10_p'] = -np.log10(p_plot)
        subset['is_lasso'] = subset['feature'].isin(lasso_selected)

        colors = np.full(len(subset), '#bdbdbd', dtype=object)
        sig_mask = subset['uni_p_value'].values < p_threshold
        risky_mask = sig_mask & (subset['log2_hr'].values > 0)
        protect_mask = sig_mask & (subset['log2_hr'].values < 0)
        colors[risky_mask] = '#d62728'
        colors[protect_mask] = '#1f77b4'

        edge_colors = np.where(subset['is_lasso'].values, 'black', 'none')
        edge_widths = np.where(subset['is_lasso'].values, 1.1, 0.0)

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            subset['log2_hr'].values,
            subset['neg_log10_p'].values,
            c=colors,
            s=55,
            alpha=0.85,
            edgecolors=edge_colors,
            linewidths=edge_widths,
            zorder=3,
        )

        ax.axhline(neg_log10_threshold, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axvline(0.0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)

        if label_top_n > 0:
            top_subset = subset.sort_values('uni_p_value').head(label_top_n)
            for _, row in top_subset.iterrows():
                ax.annotate(
                    row['feature'],
                    xy=(row['log2_hr'], row['neg_log10_p']),
                    xytext=(4, 4),
                    textcoords='offset points',
                    fontsize=7.5,
                    color='black',
                )

        ax.set_xlabel('log2(Hazard Ratio)   [protective ←    → risky]')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title(
            f'Volcano Plot: {endpoint_name} | {treatment_name}\n'
            f'(n={len(subset)} features, p-threshold = {p_threshold})'
        )
        ax.grid(alpha=0.2)

        n_sig = int(sig_mask.sum())
        n_risky = int(risky_mask.sum())
        n_protect = int(protect_mask.sum())
        n_lasso_shown = int(subset['is_lasso'].sum())

        legend_elements = [
            Line2D([0], [0], marker='o', color='#d62728', linestyle='None', markersize=7,
                   label=f'Risky & p<{p_threshold} (n={n_risky})'),
            Line2D([0], [0], marker='o', color='#1f77b4', linestyle='None', markersize=7,
                   label=f'Protective & p<{p_threshold} (n={n_protect})'),
            Line2D([0], [0], marker='o', color='#bdbdbd', linestyle='None', markersize=7,
                   label=f'Not significant (n={len(subset) - n_sig})'),
            Line2D([0], [0], marker='o', markerfacecolor='white', markeredgecolor='black',
                   linestyle='None', markersize=7, label=f'Lasso-selected (n={n_lasso_shown})'),
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=8, frameon=True)

        plt.tight_layout()
        plt.show()

        print(f'[{endpoint_name} | {treatment_name}] Volcano plot: '
              f'{len(subset)} features | significant: {n_sig} '
              f'(risky: {n_risky}, protective: {n_protect}) | lasso-selected shown: {n_lasso_shown}')


def run_univariate_analysis_and_forest_plots(cox_run_registry, coef_threshold=1e-8, p_threshold=0.05, max_features=30):
    """Run all-feature univariate Cox, compare vs. Lasso, and generate forest + volcano plots."""
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

    plot_volcano_by_treatment(
        all_uni_df,
        cox_run_registry,
        coef_threshold=coef_threshold,
        p_threshold=p_threshold,
    )

    return all_uni_df, comparison_df


# ─── IMC correlation analysis ────────────────────────────────────────────────

IMC_FILE_SPECS = [
    ('broad_clusters', 'broad_clusters_expr_density_p1.csv',
     'Broad cellular-neighborhood (15-NN) density'),
    ('local_clusters', 'local_clusters_expr_density_p1.csv',
     'Local cellular-neighborhood (5-NN) density'),
    ('celltypes', 'celltypes_density_dat_p1.csv',
     'Cell-type density'),
    ('clustered_cells', 'clustered_cells_density_dat_p1.csv',
     'Clustered cell-type density'),
]


def load_imc_density_tables(imc_dir, file_map=None, id_col='immucan_id'):
    """Load IMC density tables from `imc_dir` into a dict keyed by short name.

    `file_map` (optional) overrides the filenames, e.g.
    {'broad_clusters': 'my_custom_broad.csv'}.
    """
    imc_dir = Path(imc_dir)
    specs = IMC_FILE_SPECS
    tables = {}
    for short_name, default_filename, description in specs:
        filename = (file_map or {}).get(short_name, default_filename)
        path = imc_dir / filename
        if not path.exists():
            print(f'[IMC load] {short_name}: skipped (file not found: {path})')
            continue
        df = pd.read_csv(path)
        if id_col not in df.columns:
            print(f'[IMC load] {short_name}: skipped (no `{id_col}` column in {path.name})')
            continue
        df[id_col] = df[id_col].astype(str).str.strip()
        tables[short_name] = df
        print(f'[IMC load] {short_name}: {len(df)} rows × {df.shape[1] - 1} feature columns '
              f'({description})')
    return tables


def _imc_correlation_rows(merged_df, imc_df, cluster_cols, imc_feature_cols,
                          id_col_left, id_col_right, imc_short_name, imc_description,
                          method, min_n):
    merged_slim = merged_df[[id_col_left] + cluster_cols].copy()
    merged_slim[id_col_left] = merged_slim[id_col_left].astype(str).str.strip()
    imc_slim = imc_df[[id_col_right] + imc_feature_cols].copy()
    imc_slim[id_col_right] = imc_slim[id_col_right].astype(str).str.strip()

    joined = merged_slim.merge(
        imc_slim, left_on=id_col_left, right_on=id_col_right, how='inner'
    )
    if joined.empty:
        print(f'[IMC corr | {imc_short_name}] No patients matched on {id_col_left}↔{id_col_right}.')
        return pd.DataFrame(), 0

    rows = []
    for cluster in cluster_cols:
        for imc_feature in imc_feature_cols:
            pair = joined[[cluster, imc_feature]].apply(pd.to_numeric, errors='coerce').dropna()
            n = len(pair)
            if n < min_n or pair[cluster].nunique() < 2 or pair[imc_feature].nunique() < 2:
                rows.append({
                    'imc_table': imc_short_name,
                    'imc_description': imc_description,
                    'cluster': cluster,
                    'imc_feature': imc_feature,
                    'n': n,
                    'r': np.nan,
                    'p': np.nan,
                    'status': 'skipped_low_n_or_constant',
                })
                continue
            try:
                if method == 'pearson':
                    r, p = pearsonr(pair[cluster].values, pair[imc_feature].values)
                else:
                    r, p = spearmanr(pair[cluster].values, pair[imc_feature].values)
            except Exception as exc:
                rows.append({
                    'imc_table': imc_short_name,
                    'imc_description': imc_description,
                    'cluster': cluster,
                    'imc_feature': imc_feature,
                    'n': n,
                    'r': np.nan,
                    'p': np.nan,
                    'status': f'failed:{exc}',
                })
                continue
            rows.append({
                'imc_table': imc_short_name,
                'imc_description': imc_description,
                'cluster': cluster,
                'imc_feature': imc_feature,
                'n': int(n),
                'r': float(r),
                'p': float(p),
                'status': 'ok',
            })

    df = pd.DataFrame(rows)
    ok = df['status'] == 'ok'
    if ok.any():
        _, p_fdr, _, _ = multipletests(df.loc[ok, 'p'].values, method='fdr_bh')
        df['p_fdr'] = np.nan
        df.loc[ok, 'p_fdr'] = p_fdr
    else:
        df['p_fdr'] = np.nan

    return df, len(joined)


def correlate_clusters_with_imc(
    merged_df,
    imc_tables,
    cluster_cols=None,
    method='spearman',
    min_n=20,
    id_col_left='clinical_patient_id',
    id_col_right='immucan_id',
):
    """Correlate every H&E cluster frequency column with every IMC feature column.

    Returns a long-form DataFrame with columns:
      imc_table, imc_description, cluster, imc_feature, n, r, p, p_fdr, status.
    """
    if not imc_tables:
        print('[IMC corr] No IMC tables provided.')
        return pd.DataFrame()
    if id_col_left not in merged_df.columns:
        raise KeyError(
            f'merged_df is missing the patient-id column `{id_col_left}`. '
            f'Available columns include: {list(merged_df.columns)[:10]}'
        )
    if cluster_cols is None:
        cluster_cols = get_cluster_feature_columns(merged_df)
    if not cluster_cols:
        print('[IMC corr] No cluster frequency columns found in merged_df.')
        return pd.DataFrame()

    description_map = {short: description for short, _, description in IMC_FILE_SPECS}

    all_frames = []
    for imc_short_name, imc_df in imc_tables.items():
        imc_feature_cols = [
            c for c in imc_df.columns
            if c != id_col_right and pd.api.types.is_numeric_dtype(imc_df[c])
        ]
        if not imc_feature_cols:
            print(f'[IMC corr | {imc_short_name}] No numeric feature columns — skipping.')
            continue

        frame, n_matched = _imc_correlation_rows(
            merged_df=merged_df,
            imc_df=imc_df,
            cluster_cols=cluster_cols,
            imc_feature_cols=imc_feature_cols,
            id_col_left=id_col_left,
            id_col_right=id_col_right,
            imc_short_name=imc_short_name,
            imc_description=description_map.get(imc_short_name, ''),
            method=method,
            min_n=min_n,
        )
        if frame.empty:
            continue
        all_frames.append(frame)
        n_ok = int((frame['status'] == 'ok').sum())
        print(f'[IMC corr | {imc_short_name}] Matched {n_matched} patients; '
              f'{n_ok}/{len(frame)} correlations computed (method={method}).')

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def plot_imc_correlation_heatmap(
    corr_long_df,
    imc_short_name,
    p_threshold=0.05,
    use_fdr=True,
    cluster_order=None,
    figsize=None,
    title_suffix=None,
):
    """Plot a signed -log10(p) heatmap: rows=H&E clusters, cols=IMC features.

    Red = positive correlation, blue = negative, alpha-scaled by significance.
    Only correlations with status=='ok' are shown; non-significant cells are faded.
    """
    subset = corr_long_df[
        (corr_long_df['imc_table'] == imc_short_name)
        & (corr_long_df['status'] == 'ok')
    ].copy()
    if subset.empty:
        print(f'[IMC heatmap | {imc_short_name}] No valid correlations to plot.')
        return

    p_col = _select_p_col(subset, use_fdr)
    subset['signed_neg_log10_p'] = -np.log10(subset[p_col].clip(lower=1e-300)) * np.sign(subset['r'])

    pivot = subset.pivot(index='cluster', columns='imc_feature', values='signed_neg_log10_p')

    if cluster_order is not None:
        pivot = pivot.reindex([c for c in cluster_order if c in pivot.index])
    else:
        pivot = pivot.reindex(sorted(pivot.index, key=_cluster_sort_key))

    pivot = pivot[sorted(pivot.columns)]

    n_rows, n_cols = pivot.shape
    if figsize is None:
        figsize = (max(6, 0.35 * n_cols + 3), max(4, 0.3 * n_rows + 2))

    abs_max = float(np.nanmax(np.abs(pivot.values))) if np.isfinite(pivot.values).any() else 1.0
    vmax = max(abs_max, 1.0)

    sig_pivot = subset.pivot(index='cluster', columns='imc_feature', values=p_col)
    sig_pivot = sig_pivot.reindex(index=pivot.index, columns=pivot.columns)
    annot = np.where(sig_pivot.values < p_threshold, '*', '')

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.3,
        linecolor='white',
        cbar_kws={'label': f'signed -log10({p_col})'},
        annot=annot,
        fmt='',
        annot_kws={'fontsize': 8, 'color': 'black'},
    )
    title = f'H&E cluster ↔ IMC correlation: {imc_short_name}'
    if title_suffix:
        title += f' — {title_suffix}'
    title += f'\n(* = {p_col} < {p_threshold})'
    ax.set_title(title)
    ax.set_xlabel('IMC feature')
    ax.set_ylabel('H&E cluster')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_imc_correlation_r_heatmap(
    corr_long_df,
    imc_short_name,
    p_threshold=0.05,
    use_fdr=True,
    cluster_order=None,
    figsize=None,
    title_suffix=None,
):
    """Heatmap of plain Spearman correlation coefficients (r in [-1, 1]).

    Cells significant at p (or FDR) < threshold are annotated with '*'.
    Companion to `plot_imc_correlation_heatmap`, which encodes significance
    in color rather than coefficient strength.
    """
    subset = corr_long_df[
        (corr_long_df['imc_table'] == imc_short_name)
        & (corr_long_df['status'] == 'ok')
    ].copy()
    if subset.empty:
        print(f'[IMC r-heatmap | {imc_short_name}] No valid correlations to plot.')
        return

    p_col = _select_p_col(subset, use_fdr)

    pivot_r = subset.pivot(index='cluster', columns='imc_feature', values='r')
    if cluster_order is not None:
        pivot_r = pivot_r.reindex([c for c in cluster_order if c in pivot_r.index])
    else:
        pivot_r = pivot_r.reindex(sorted(pivot_r.index, key=_cluster_sort_key))
    pivot_r = pivot_r[sorted(pivot_r.columns)]

    n_rows, n_cols = pivot_r.shape
    if figsize is None:
        figsize = (max(6, 0.35 * n_cols + 3), max(4, 0.3 * n_rows + 2))

    sig_pivot = subset.pivot(index='cluster', columns='imc_feature', values=p_col)
    sig_pivot = sig_pivot.reindex(index=pivot_r.index, columns=pivot_r.columns)
    annot = np.where(sig_pivot.values < p_threshold, '*', '')

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_r,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.3,
        linecolor='white',
        cbar_kws={'label': 'Spearman r'},
        annot=annot,
        fmt='',
        annot_kws={'fontsize': 8, 'color': 'black'},
    )
    title = f'Spearman r heatmap: H&E cluster ↔ IMC ({imc_short_name})'
    if title_suffix:
        title += f' — {title_suffix}'
    title += f'\n(* = {p_col} < {p_threshold})'
    ax.set_title(title)
    ax.set_xlabel('IMC feature')
    ax.set_ylabel('H&E cluster')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_imc_correlation_bubbles(
    corr_long_df,
    imc_short_name,
    p_threshold=0.05,
    use_fdr=True,
    cluster_order=None,
    figsize=None,
    title_suffix=None,
    min_marker_size=15,
    max_marker_size=320,
):
    """Bubble plot — color = Spearman r, size = -log10(p-value).

    Combines direction/strength (color in [-1, 1]) and significance (dot size)
    in a single panel. Significance threshold is shown as a dashed-edge ring.
    """
    subset = corr_long_df[
        (corr_long_df['imc_table'] == imc_short_name)
        & (corr_long_df['status'] == 'ok')
    ].copy()
    if subset.empty:
        print(f'[IMC bubbles | {imc_short_name}] No valid correlations to plot.')
        return

    p_col = _select_p_col(subset, use_fdr)
    subset['neg_log10_p'] = -np.log10(subset[p_col].clip(lower=1e-300))

    if cluster_order is not None:
        clusters = [c for c in cluster_order if c in subset['cluster'].unique()]
    else:
        clusters = sorted(subset['cluster'].unique(), key=_cluster_sort_key)
    imc_features = sorted(subset['imc_feature'].unique())

    cluster_to_y = {c: i for i, c in enumerate(clusters)}
    feature_to_x = {f: i for i, f in enumerate(imc_features)}
    subset['x'] = subset['imc_feature'].map(feature_to_x)
    subset['y'] = subset['cluster'].map(cluster_to_y)

    nlp_max = float(np.nanmax(subset['neg_log10_p'].values))
    if not np.isfinite(nlp_max) or nlp_max <= 0:
        nlp_max = 1.0
    sizes = min_marker_size + (subset['neg_log10_p'].values / nlp_max) * (max_marker_size - min_marker_size)
    sizes = np.clip(sizes, min_marker_size, max_marker_size)

    n_rows = len(clusters)
    n_cols = len(imc_features)
    if figsize is None:
        figsize = (max(7, 0.42 * n_cols + 3.5), max(4.5, 0.38 * n_rows + 2))

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        subset['x'].values,
        subset['y'].values,
        c=subset['r'].values,
        s=sizes,
        cmap='RdBu_r',
        vmin=-1.0,
        vmax=1.0,
        edgecolors='black',
        linewidths=0.4,
        alpha=0.95,
    )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(imc_features, rotation=45, ha='right')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(clusters)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()
    ax.grid(alpha=0.2)

    ax.set_xlabel('IMC feature')
    ax.set_ylabel('H&E cluster')
    title = f'Bubble plot: H&E cluster ↔ IMC ({imc_short_name})'
    if title_suffix:
        title += f' — {title_suffix}'
    title += f'\n(color = Spearman r, size = -log10({p_col}))'
    ax.set_title(title)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label('Spearman r')

    legend_p_levels = [p_threshold, p_threshold / 10.0, p_threshold / 100.0]
    legend_handles = []
    legend_labels = []
    for p_level in legend_p_levels:
        nlp = -math.log10(p_level)
        size = min_marker_size + min(nlp / nlp_max, 1.0) * (max_marker_size - min_marker_size)
        legend_handles.append(plt.scatter([], [], s=size, c='lightgrey',
                                          edgecolors='black', linewidths=0.4))
        legend_labels.append(f'{p_col} = {p_level:g}')
    ax.legend(
        legend_handles, legend_labels,
        scatterpoints=1, frameon=True,
        loc='upper left', bbox_to_anchor=(1.18, 1.0),
        title='Significance',
        labelspacing=1.4,
    )

    plt.tight_layout()
    plt.show()


def summarize_top_imc_correlations(corr_long_df, top_n=20, use_fdr=True):
    """Print and return the top-N most significant cluster × IMC pairs (per IMC table)."""
    if corr_long_df.empty:
        print('[IMC top pairs] No correlations available.')
        return pd.DataFrame()

    p_col = _select_p_col(corr_long_df, use_fdr)
    ok = corr_long_df[corr_long_df['status'] == 'ok'].copy()
    if ok.empty:
        print('[IMC top pairs] No valid correlations.')
        return pd.DataFrame()

    ok = ok.sort_values([p_col, 'p'], ascending=True)
    top_rows = []
    for imc_short_name, group in ok.groupby('imc_table'):
        head = group.head(top_n).copy()
        top_rows.append(head)
        print(f'\n[IMC top pairs | {imc_short_name}] Top {len(head)} by {p_col}:')
        display_cols = ['cluster', 'imc_feature', 'r', 'p', p_col, 'n']
        display(head[display_cols])
    return pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()


def run_imc_correlation_analysis(
    merged_df,
    imc_dir,
    method='spearman',
    p_threshold=0.05,
    use_fdr=True,
    top_n=20,
    min_n=20,
    id_col_left='clinical_patient_id',
    id_col_right='immucan_id',
    file_map=None,
):
    """Top-level orchestrator:
      1. Load IMC tables from `imc_dir`.
      2. Correlate each H&E cluster frequency with each IMC feature.
      3. Per IMC table, draw three views:
           a) signed -log10(p) heatmap (significance-colored)
           b) Spearman r heatmap (direction & strength only)
           c) bubble plot (color = r, size = -log10(p))
      4. Print/return the top-N most significant pairs per table.
    """
    imc_tables = load_imc_density_tables(imc_dir, file_map=file_map, id_col=id_col_right)
    if not imc_tables:
        print('[IMC analysis] No IMC tables loaded; aborting.')
        return pd.DataFrame(), pd.DataFrame()

    corr_df = correlate_clusters_with_imc(
        merged_df=merged_df,
        imc_tables=imc_tables,
        method=method,
        min_n=min_n,
        id_col_left=id_col_left,
        id_col_right=id_col_right,
    )
    if corr_df.empty:
        return corr_df, pd.DataFrame()

    for imc_short_name in imc_tables.keys():
        plot_imc_correlation_heatmap(
            corr_long_df=corr_df,
            imc_short_name=imc_short_name,
            p_threshold=p_threshold,
            use_fdr=use_fdr,
        )
        plot_imc_correlation_r_heatmap(
            corr_long_df=corr_df,
            imc_short_name=imc_short_name,
            p_threshold=p_threshold,
            use_fdr=use_fdr,
        )
        plot_imc_correlation_bubbles(
            corr_long_df=corr_df,
            imc_short_name=imc_short_name,
            p_threshold=p_threshold,
            use_fdr=use_fdr,
        )

    top_df = summarize_top_imc_correlations(
        corr_long_df=corr_df,
        top_n=top_n,
        use_fdr=use_fdr,
    )
    return corr_df, top_df
