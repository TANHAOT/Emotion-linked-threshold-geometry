#!/usr/bin/env python3
"""Summarize SAE activation and top-10 frequency across unfairness levels."""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--subjects', default='400-419')
    p.add_argument('--output', type=Path, default=Path('unfairness_feature_screening.csv'))
    a=p.parse_args()
    lo,hi=map(int,a.subjects.split('-'))
    rows=[]; top_counts=defaultdict(Counter); totals=Counter()
    for subj in range(lo,hi+1):
        settings=json.loads((a.inputs/f'{subj}_game_setting_prompt.json').read_text())
        fmap={i:int(15-r['amount_of_allocation']) for i,r in enumerate(settings)}
        feat=a.results/f'sae_features_{subj}.csv'
        out=a.results/f'output_{subj}.txt'
        if feat.exists():
            df=pd.read_csv(feat)
            df['unfairness']=df['round'].map(fmap)
            for (u,fid),g in df.groupby(['unfairness','feature_id']):
                rows.append({'subject':subj,'unfairness':u,'feature_id':int(fid),'nonzero_mean':g['value'].mean(),'nonzero_count':len(g)})
        if out.exists():
            odf=pd.read_csv(out, sep='\t')
            for i,r in odf.iterrows():
                u=fmap.get(i)
                ids=[x for x in str(r.get('Top_10_SAE_Feature_Last','')).split('|') if x and x!='nan']
                top_counts[u].update(ids); totals[u]+=1
    act=pd.DataFrame(rows)
    if act.empty: raise SystemExit('No SAE feature files found.')
    agg=act.groupby(['unfairness','feature_id']).agg(mean_nonzero_activation=('nonzero_mean','mean'),subjects_nonzero=('subject','nunique')).reset_index()
    freq=[]
    for u,c in top_counts.items():
        for fid,n in c.items(): freq.append({'unfairness':u,'feature_id':int(fid),'top10_frequency':n/max(totals[u],1)})
    freq=pd.DataFrame(freq)
    merged=agg.merge(freq,on=['unfairness','feature_id'],how='outer').fillna(0)
    summary=merged.groupby('feature_id').agg(
        activation_range=('mean_nonzero_activation', lambda x: x.max()-x.min()),
        top10_frequency_range=('top10_frequency', lambda x: x.max()-x.min()),
        max_top10_frequency=('top10_frequency','max'),
    ).reset_index()
    summary['screen_score']=summary['activation_range'].rank(pct=True)+summary['top10_frequency_range'].rank(pct=True)
    summary=summary.sort_values('screen_score', ascending=False)
    a.output.parent.mkdir(parents=True,exist_ok=True); summary.to_csv(a.output,index=False)
    print(summary.head(30).to_string(index=False))
    print('Supplement-reported pilot candidates: 9979, 10018, 11424, 16217; formal steering target: 16217')
if __name__=='__main__': main()
