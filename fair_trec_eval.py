#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#%%
def DCG(df_query, k=10):
    df = df_query.copy()
    df['rank'] += 1 #TODO: Move to generating file
    df['DCG'] = df['relevance'] / np.log2(1+df['rank']) * (df['rank']).between(1,k, inclusive='both') #NOTE! 
    return df.groupby('query')['DCG'].sum()

def iDCG(df_query, k=10):
    df = df_query.copy()
    df['ideal_rank'] = df.groupby('query')['relevance'].rank('first', ascending=False)
    df['IDCG'] = df['relevance'] / np.log2(1+df['ideal_rank'])  * (df['ideal_rank']).between(1,k, inclusive='both') #NOTE!
    return df.groupby('query')['IDCG'].sum()

def nDCG(df_query, k=10):
    df = pd.concat([DCG(df_query, k), iDCG(df_query, k)],axis=1)
    df[f'nDCG@{k}'] = df['DCG'] / df['IDCG']
    return df[[f'nDCG@{k}']].fillna(0)

def precision(df_query):
    df = df_query.copy()
    df['TP'] = df.groupby('query')['rel_binary'].cumsum()
    df['precision'] = df['TP'] / df['rank']
    return df[['query', 'TP', 'rank', 'precision']]

def AP(df_query, df_qrels):
    df = df_query.merge(relevant_docs(df_qrels), on='query', how='left')
    df['TP'] = df.groupby('query')['rel_binary'].cumsum()
    df['marginal_precision'] = df['TP'] * df['rel_binary'] / (df['rank'] * df['relevant_docs'])
    return df.groupby('query')['marginal_precision'].sum()

def recall(df_query, df_qrels):
    df = df_query.merge(relevant_docs(df_qrels), on='query', how='left')
    df['TP'] = df.groupby('query')['rel_binary'].cumsum()
    df['recall'] = df['TP'] / df['relevant_docs']
    return df[['query', 'TP', 'rank', 'recall']]

def relevant_docs(df_qrels):
    df_positives = df_qrels.groupby('query').sum().reset_index().rename({'rel_binary' : 'relevant_docs'}, axis=1)
    return df_positives[['query', 'relevant_docs']]

def recip_rank(df_query):
    df=df_query.loc[
        df_query['rel_binary'] == 1
    ].groupby(
        'query'
    )['rank'].min().reset_index()
    df['recip_rank'] = 1 / df['rank']
    return df[['query', 'recip_rank']]

def precision_at(df_precision, k):
    return df_precision.loc[df_precision['rank'] == k]

# %%
# Fairness metrics, categorized:
# - Exposure vs Engagement (Do we need user feedback?)
# - Attention distribution (Flat vs Exponential vs Geometric)
# - Merit modeling (Flat vs Engagement vs Relevance)

# At first, don't apply distance function, just take the different values, and then we can make a distance function later. 
# - Distance function (Abs vs square vs KL-divergence vs maximin)

def create_super_file(experiment_name, run_name):
    df_qrels=pd.read_csv(f'runs/{experiment_name}/qrels_{run_name}.txt', sep=' ', names=['query', 'unknown', 'document', 'relevance'])
    df_run = pd.read_csv(f'runs/{experiment_name}/{run_name}.run', sep=' ', names=['query', 'unknown', 'document', 'rank', 'score', 'experiment', 'true_relevance', 'clicks'])
    df_group = pd.read_csv(f'runs/{experiment_name}/groups_{run_name}.txt', sep=' ', names=['document', 'lean', 'group'])
    df_user = pd.read_csv(f'runs/{experiment_name}/users_{run_name}.txt', sep=' ', names=['query', 'lean', 'openness', 'ranker'])
    df_user['susceptibility'] = df_user['openness']/2 - np.random.random(df_user.shape[0])/10

    df = df_run.merge(
            df_qrels, 
            on=['document', 'query'], 
            how='left'
        ).fillna(
            {'relevance' : 0, 'rel_binary' : 0}
        ).merge(
            df_group, 
            on='document', 
            how='left'
        ).merge(
            df_user, 
            on='query', 
            how='left',
            suffixes=['_doc', '_user']
        ).sort_values(by=['query', 'rank'])


    df['merit'] = df['relevance']
    df['exposure'] = df['rank'].apply(lambda x: 1/np.log2(x+2))
    df['unfairness'] = df['group'] * df['exposure']
    df['click_unfairness'] = df['group'] * df['clicks']
    df['impact'] = df['openness'] * df['unfairness']
    df['bias_side'] = 0
    df.loc[df['ranker'] == 'right', 'bias_side'] = 1
    df.loc[df['ranker'] == 'left', 'bias_side'] = -1

    df['impact2'] = df['bias_side'] * df['susceptibility']
    df['error'] = abs(df['true_relevance'] - df['score'])

    return df, df_user
    
def graph(experiment_name, save=False, seeds=range(10000000000)):
    runs = [file_name for file_name in os.listdir(f'runs/{experiment_name}/') if file_name.endswith('.run')]   
    dfs=[]
    for file_name in runs:
        run_name = file_name.split('.')[0]
        seed, run_type= run_name.split('_')
        if int(seed) not in seeds:
            continue
        print(file_name)
        df, df_user = create_super_file(experiment_name, run_name)    
        df_per_query = df.groupby(
            ['query', 'group']
        ).mean(
            numeric_only=True
        ).reset_index(
        ).merge(
            nDCG(df, 5), on='query'
        ).merge(
            nDCG(df, 1), on='query'
        ).merge(
            nDCG(df, 10), on='query'
        ).merge(
            nDCG(df, 30), on='query'
        ).set_index(
            ['query', 'group']
        )
        df_plot = (df_per_query.groupby(['group']).cumsum() / (df_per_query!='FILLER_VALUE').groupby(['group']).cumsum())
        df_plot['unfairness_per_merit'] = df_plot['unfairness'] / df_plot['merit']
        df_plot['click_unfairness_per_merit'] = df_plot['click_unfairness'] / df_plot['merit']
        df_plot=df_plot.groupby('query').sum().merge(df_user[['query', 'ranker']], on='query')
        df_plot['unfairness_per_merit'] = abs(df_plot['unfairness_per_merit'])
        df_plot['click_unfairness_per_merit'] = abs(df_plot['click_unfairness_per_merit'])
        df_plot['nDCG@5'] /= 2 #FIXME
        df_plot['nDCG@1'] /= 2 #FIXME
        df_plot['nDCG@10'] /= 2 #FIXME   
        df_plot['nDCG@30'] /= 2 #FIXME
        df_plot['query_no'] = df_plot['query'].str[-5:].astype(int)

        dfs.append(df_plot.assign(seed=seed, run_type=run_type))
    df_out = pd.concat(dfs, axis=0)
    df_out['run_type'].replace(to_replace={
            'baseline' : 'D-ULTR(Glob)',
        }, inplace=True) 
    if experiment_name == 'labda_test':
        df_out['run_type'].replace(to_replace={
            'lambda-005' : 'lambda = 0.005',
            'lambda-001' : 'lambda = 0.001',
            'lambda-02' : 'lambda = 0.02' ,
            'lambda-01' : 'lambda = 0.01',
            'lambda-003' : 'lambda = 0.003',
            'lambda-05' : 'lambda = 0.05' 

        }, inplace=True) 
    color_list = ['black', 'yellow', 'orange', 'red', 'blue', 'green']
    gen_colors = (color for color in color_list)

    df_avg = df_out.loc[df_out['query_no'] > 100].copy()
    df_avg['unfairness'] = abs(df_avg['unfairness'])
    df_avg['Users'] = df_avg['query_no']

    grouped = df_avg.groupby(['run_type', 'Users']).mean().reset_index().groupby('run_type')
    fig, axs = plt.subplots(1,3,figsize=[15,5])
    fig.suptitle(f'Overall Performance')
    fig.set_dpi(1000.0)
    colors = {}
    for key, group in grouped:
        colors[key] = next(gen_colors)
        group.plot(ax=axs[0], kind='line', x='Users', y='nDCG@10', label=key, color=colors[key], title='System Relevance')
        group.plot(ax=axs[2], kind='line', x='Users', y='impact', label=key, color=colors[key], title='Electoral Impact', legend=None)
        group.plot(ax=axs[1], kind='line', x='Users', y='unfairness_per_merit', label=key, color=colors[key], title='Exposure Inequity', legend=None)

    if save:
        plt.savefig(f'graphs/{experiment_name}.png', facecolor=(1, 1, 1))

    # grouped = df_avg.groupby(['run_type', 'query_no']).mean().reset_index().groupby('run_type')
    # fig, axs = plt.subplots(2,2, figsize=[12,12])
    # fig.suptitle(f'Clicks Performance')
    # for key, group in grouped:
    #     group.plot(ax=axs[0,0], kind='line', x='query_no', y='impact2', label=key, color=colors[key], title='impact2')
    #     group.plot(ax=axs[0,1], kind='line', x='query_no', y='click_unfairness', label=key, color=colors[key], title='Clicks Exposure Inequality', legend=None)
    #     group.plot(ax=axs[1,0], kind='line', x='query_no', y='impact', label=key, color=colors[key], title='impact', legend=None)
    #     group.plot(ax=axs[1,1], kind='line', x='query_no', y='click_unfairness_per_merit', label=key, color=colors[key], title='Clicks Exposure per Merit Inequality', legend=None)
    # plt.show()
    # if save:
    #     plt.savefig(f'graphs/clicks_{experiment_name}.png', facecolor=(1, 1, 1))


    fig, axs = plt.subplots(1,4, figsize=[20,5])
    fig.set_dpi(1000.0)
    for key, group in grouped:
        group.plot(ax=axs[0], kind='line', x='Users', y='nDCG@1', label=key, color=colors[key], title='nDCG@1')
        group.plot(ax=axs[1], kind='line', x='Users', y='nDCG@5', label=key, color=colors[key], title='nDCG@5', legend=None)
        group.plot(ax=axs[2], kind='line', x='Users', y='nDCG@10', label=key, color=colors[key], title='nDCG@10', legend=None)
        group.plot(ax=axs[3], kind='line', x='Users', y='nDCG@30', label=key, color=colors[key], title='nDCG@30', legend=None)
    if save:
        plt.savefig(f'graphs/nDCG{experiment_name}.png', facecolor=(1, 1, 1))



#%%
if __name__ == '__main__':
    experiment_name = ''
    if experiment_name != 'skew_pop':
        graph(experiment_name, save=True)
    else:
        seeds = range(10000000000)
        runs = [file_name for file_name in os.listdir(f'runs/{experiment_name}/') if file_name.endswith('.run')]   
        dfs=[]
        for file_name in runs:
            run_name = file_name.split('.')[0]
            seed, run_type= run_name.split('_')
            if int(seed) not in seeds: 
                continue
            print(file_name)
            system = ''.join(letter for letter in run_type if not letter.isnumeric())
            p_left = ''.join(letter for letter in run_type if letter.isnumeric())
            df, df_user = create_super_file(experiment_name, run_name)    
            df_per_query = df.groupby(
                ['query', 'group']
            ).mean(numeric_only=True
            ).reset_index(
            ).merge(
                nDCG(df, 5), on='query'
            ).merge(
                nDCG(df, 1), on='query'
            ).merge(
                nDCG(df, 10), on='query'
            ).merge(
                nDCG(df, 30), on='query'
            ).set_index(
                ['query', 'group']
            )
            df_plot = (df_per_query.groupby(['group']).cumsum() / (df_per_query!='FILLER_VALUE').groupby(['group']).cumsum())
            df_plot['unfairness_per_merit'] = df_plot['unfairness'] / df_plot['merit']

            df_plot['click_unfairness_per_merit'] = df_plot['click_unfairness'] / df_plot['merit']
            df_plot=df_plot.groupby('query').sum().merge(df_user[['query', 'ranker']], on='query')
            df_plot['click_unfairness_per_merit'] = abs(df_plot['click_unfairness_per_merit'])
            df_plot['nDCG@5'] /= 2 #FIXME
            df_plot['nDCG@1'] /= 2 #FIXME
            df_plot['nDCG@10'] /= 2 #FIXME   
            df_plot['nDCG@30'] /= 2 #FIXME
            df_plot['query_no'] = df_plot['query'].str[-5:].astype(int)

            dfs.append(df_plot.loc[df_plot['query_no'] == 2999].assign(seed=seed, system=system, p_left = p_left))
        df_out = pd.concat(dfs, axis=0)
        df_out.to_csv(f'{experiment_name}.csv')
        df_out=pd.read_csv(f'{experiment_name}.csv')

        color_list = ['black', 'red', 'blue', 'green', 'yellow', 'orange']
        gen_colors = (color for color in color_list)

        df_avg = df_out.loc[df_out['query_no'] > 100].copy()
        df_avg['unfairness'] = abs(df_avg['unfairness'])
        df_avg['unfairness_per_merit'] = abs(df_avg['unfairness_per_merit'])

        df_avg['Proportion of left-wing users'] = df_avg['p_left']
        grouped = df_avg.groupby(['system', 'Proportion of left-wing users']).mean().reset_index().groupby('system')
        fig, axs = plt.subplots(1,3,figsize=[15,5])
        fig.suptitle(f'Overall Performance')
        fig.set_dpi(1000.0)
        colors = {}
        for key, group in grouped:
            colors[key] = next(gen_colors)
            group.plot(ax=axs[0], kind='line', x='Proportion of left-wing users', y='nDCG@10', label=key, color=colors[key], title='System Relevance')
            group.plot(ax=axs[2], kind='line', x='Proportion of left-wing users', y='impact', label=key, color=colors[key], title='Electoral Impact', legend=None)
            group.plot(ax=axs[1], kind='line', x='Proportion of left-wing users', y='unfairness_per_merit', label=key, color=colors[key], title='Amortized Inequity', legend=None)
        plt.savefig(f'graphs/{experiment_name}.png', facecolor=(1, 1, 1))

    # fig, axs = plt.subplots(2,2, figsize=[12,12])
    # fig.suptitle(f'Clicks Performance')
    # for key, group in grouped:
    #     group.plot(ax=axs[0,0], kind='line', x='p_left', y='impact2', label=key, color=colors[key], title='impact2')
    #     group.plot(ax=axs[0,1], kind='line', x='p_left', y='click_unfairness', label=key, color=colors[key], title='Clicks Exposure Inequality', legend=None)
    #     group.plot(ax=axs[1,0], kind='line', x='p_left', y='impact', label=key, color=colors[key], title='impact', legend=None)
    #     group.plot(ax=axs[1,1], kind='line', x='p_left', y='click_unfairness_per_merit', label=key, color=colors[key], title='Clicks Exposure per Merit Inequality', legend=None)
    # plt.savefig(f'graphs/clicks{experiment_name}.png', facecolor=(1, 1, 1))

    
    # fig, axs = plt.subplots(2,2, figsize=[12,12])
    # fig.suptitle('nDCG')
    # for key, group in grouped:
    #     group.plot(ax=axs[0,0], kind='line', x='p_left', y='nDCG@1', label=key, color=colors[key], title='nDCG@1')
    #     group.plot(ax=axs[0,1], kind='line', x='p_left', y='nDCG@5', label=key, color=colors[key], title='nDCG@5', legend=None)
    #     group.plot(ax=axs[1,0], kind='line', x='p_left', y='nDCG@10', label=key, color=colors[key], title='nDCG@10', legend=None)
    #     group.plot(ax=axs[1,1], kind='line', x='p_left', y='nDCG@30', label=key, color=colors[key], title='nDCG@30', legend=None)
    # plt.savefig(f'graphs/nDCG{experiment_name}.png', facecolor=(1, 1, 1))



# fig, axs = plt.subplots(2,2, figsize=[12,12])
# for key, group in grouped:
#     group.plot(ax=axs[0,0], kind='line', x='query_no', y='nDCG@1', label=key, color=colors[key], title='nDCG@1')
#     group.plot(ax=axs[0,1], kind='line', x='query_no', y='nDCG@5', label=key, color=colors[key], title='nDCG@5', legend=None)
#     group.plot(ax=axs[1,0], kind='line', x='query_no', y='nDCG@10', label=key, color=colors[key], title='nDCG@10', legend=None)
#     group.plot(ax=axs[1,1], kind='line', x='query_no', y='nDCG@30', label=key, color=colors[key], title='nDCG@30', legend=None)

# for seed in range(302, 412, 10):
#     grouped=df_out.loc[df_out['seed'] == seed].groupby('run_type')

#     fig, ax = plt.subplots()
#     for key, group in grouped:
#         group.plot(ax=ax, kind='line', x='query_no', y='impact', label=key, color=colors[key], title=seed)
#     plt.show()
6# %%
# if run_type == 'mixed':
#     df_per_query = df_per_query.merge(df_user, on='query')
#     df_per_query['query_no'] = df_per_query['query'].str[-5:].astype(int)
#     fig, ax = plt.subplots()
#     colors = {'right' : 'red', 'left' : 'blue', 'relevance' : 'black'}
#     grouped = df_per_query.groupby('ranker')
#     for key, group in grouped:
#         group.plot(ax=ax, kind='scatter', x='query_no', y='unfairness', label=key, color=colors[key], title=run_name)
#     plt.show()

# p = 0.05
# k = 30
# attention_dict = {
#         'log' : lambda x: 1/np.log2(x+2) if x <= k else 0,
#         'flat': lambda x: 1 if x <= k else 0,
#         'exponential' : lambda x: (1-p) ** x * p if x <= k else 0
#     }
# attention_function = attention_dict['log']

# df_plot[300:].plot(x='query_no', y=['unfairness', 'impact'], title=run_name)
# df_plot[300:].plot(x='query_no', y=['nDCG'], title=run_name)
# %%
