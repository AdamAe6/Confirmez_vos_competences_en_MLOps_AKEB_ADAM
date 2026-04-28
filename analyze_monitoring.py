"""
analyze_monitoring.py

Script d'analyse des logs de production (ETAPE 3)
- lit la collection `predictions` depuis MongoDB (MONGO_URI dans .env)
- calcule des métriques globales et journalières
- détecte un drift simple (early vs late) et (si scipy présent) calcule KS p-value
- exporte : monitoring_report.json, monitoring_summary.csv, production_logs_analysis.csv

Usage:
  python analyze_monitoring.py --limit 10000

"""
import os
import json
import argparse
from datetime import datetime
import logging

import pandas as pd
import numpy as np
from dotenv import load_dotenv

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

try:
    from scipy.stats import ks_2samp
except Exception:
    ks_2samp = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyze_monitoring")


def connect_mongo(uri, server_timeout=5000):
    if MongoClient is None:
        raise ImportError("pymongo is required to connect to MongoDB. Install with: pip install pymongo")
    client = MongoClient(uri, serverSelectionTimeoutMS=server_timeout)
    client.admin.command('ping')
    return client


def load_from_mongo(mongo_uri, db_name=None, collection_name='predictions', limit=10000):
    client = connect_mongo(mongo_uri)
    # determine db_name from uri if not provided
    if db_name is None:
        # try to extract after last '/'
        if '/' in mongo_uri:
            possible = mongo_uri.rsplit('/', 1)[-1].split('?')[0]
            if possible:
                db_name = possible
    if not db_name:
        db_name = 'perso'

    db = client[db_name]
    coll = db[collection_name]
    docs = list(coll.find().limit(limit).sort('timestamp', -1))
    if not docs:
        return None

    df = pd.DataFrame(docs)
    # drop _id for analysis
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])

    # normalize timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['timestamp'] = pd.NaT

    # ensure numeric
    for col in ['probability', 'execution_time_ms', 'prediction']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # risk level
    if 'probability' in df.columns:
        df['risk_level'] = df['probability'].apply(lambda x: 'BAS' if x < 0.3 else 'MOYEN' if x < 0.5 else 'ÉLEVÉ')
    else:
        df['risk_level'] = 'N/A'

    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour

    return df


def compute_global_stats(df):
    total = len(df)
    prob_mean = float(df['probability'].mean()) if 'probability' in df.columns else None
    prob_std = float(df['probability'].std()) if 'probability' in df.columns else None
    latency_mean = float(df['execution_time_ms'].mean()) if 'execution_time_ms' in df.columns else None
    latency_p99 = float(np.percentile(df['execution_time_ms'].dropna(), 99)) if 'execution_time_ms' in df.columns else None
    high_risk = int((df['probability'] >= 0.5).sum()) if 'probability' in df.columns else None
    errors = int((df['status'] == 'error').sum()) if 'status' in df.columns else 0

    stats = {
        'total_predictions': int(total),
        'probability_mean': prob_mean,
        'probability_std': prob_std,
        'latency_mean_ms': latency_mean,
        'latency_p99_ms': latency_p99,
        'high_risk_count': high_risk,
        'error_count': errors
    }
    return stats


def detect_drift(df):
    # simple early vs late split
    if len(df) < 4 or 'probability' not in df.columns:
        return {'drift_detected': False, 'mean_diff': None, 'ks_pvalue': None}

    mid = len(df) // 2
    early = df.iloc[:mid]['probability'].dropna()
    late = df.iloc[mid:]['probability'].dropna()
    mean_diff = abs(float(early.mean()) - float(late.mean()))
    drift_flag = mean_diff > 0.05
    ks_p = None
    if ks_2samp is not None and len(early) > 0 and len(late) > 0:
        try:
            ks_res = ks_2samp(early, late)
            ks_p = float(ks_res.pvalue)
        except Exception:
            ks_p = None

    return {'drift_detected': bool(drift_flag), 'mean_diff': float(mean_diff), 'ks_pvalue': ks_p}


def daily_summary(df):
    if 'date' not in df.columns:
        return pd.DataFrame()
    agg = df.groupby('date').agg(
        total_predictions=('probability', 'count'),
        probability_mean=('probability', 'mean'),
        probability_std=('probability', 'std'),
        latency_mean_ms=('execution_time_ms', 'mean'),
        latency_p99_ms=('execution_time_ms', lambda x: np.percentile(x.dropna(), 99) if len(x.dropna())>0 else None),
        high_risk_count=('probability', lambda x: int((x>=0.5).sum()))
    ).reset_index()
    return agg


def generate_reports(df, out_dir='.'):
    os.makedirs(out_dir, exist_ok=True)
    global_stats = compute_global_stats(df)
    drift = detect_drift(df)
    summary_df = daily_summary(df)

    report = {
        'generated_at': datetime.utcnow().isoformat(),
        'global_stats': global_stats,
        'drift': drift,
        'daily_summary_rows': int(len(summary_df))
    }

    # save report JSON
    report_path = os.path.join(out_dir, 'monitoring_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # save summary CSV
    summary_path = os.path.join(out_dir, 'monitoring_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    # save enriched logs
    analysis_path = os.path.join(out_dir, 'production_logs_analysis.csv')
    df.to_csv(analysis_path, index=False)

    return {'report': report_path, 'summary': summary_path, 'analysis': analysis_path}


def main():
    parser = argparse.ArgumentParser(description='Analyze production logs from MongoDB')
    parser.add_argument('--mongo-uri', type=str, default=None, help='MongoDB URI (overrides .env)')
    parser.add_argument('--db', type=str, default=None, help='Database name (optional)')
    parser.add_argument('--collection', type=str, default='predictions', help='Collection name')
    parser.add_argument('--limit', type=int, default=10000, help='Max documents to load')
    parser.add_argument('--out', type=str, default='.', help='Output directory for reports')

    args = parser.parse_args()

    load_dotenv()
    mongo_uri = args.mongo_uri or os.getenv('MONGO_URI')
    if not mongo_uri:
        logger.error('MONGO_URI not provided in args or .env')
        return

    try:
        logger.info('Loading data from MongoDB...')
        df = load_from_mongo(mongo_uri, db_name=args.db, collection_name=args.collection, limit=args.limit)
        if df is None or len(df) == 0:
            logger.warning('No documents found in MongoDB collection')
            return

        logger.info(f'Loaded {len(df)} documents')

        out = generate_reports(df, out_dir=args.out)
        logger.info('Reports generated:')
        for k, v in out.items():
            logger.info(f'  {k}: {v}')

    except Exception as e:
        logger.exception(f'Error during analysis: {e}')


if __name__ == '__main__':
    main()
