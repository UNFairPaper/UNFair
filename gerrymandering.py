# %%
from typing import Optional
from dataclasses import dataclass
from numpy import random
import os
import numpy as np
import pandas as pd

from fair_trec_eval import graph
# %%
# Definitions
@dataclass
class User:
    name: str
    lean: float
    susceptibility: float
    ranker: str


@dataclass
class Document:
    name: str
    lean: float
    relevance: float
    clicks: float
    exposure: float
    pred_relevance: float
    group: Optional[int]


def baseline_ranker(document: Document):
    '''D-Ultr(Glob) in Morik and Singh (2020)'''
    return document.pred_relevance


def oracle(document: Document):
    '''Ranker with access to true relevance; optimum non user-based performance'''
    return document.relevance


def worst_case(document: Document):
    '''Worst possible ranker, minimum non user-based performance '''
    return -document.relevance


def random_ranker(_):
    '''Random ranker, to establish possible range of performance of rankers'''
    return random.random()

def naive_ranker(document: Document):
    '''Naive in Morik and Singh (2020)'''
    return document.clicks


def generate_users(p_left: float = 0.5, n_users: int = 3000) -> list[User]:
    return [
        User(
            name=f'User{str(i).zfill(5)}',
            lean=np.clip(random.normal(0.5, 0.2), -1, 1) *
            random.choice([-1, 1], p=[p_left, 1-p_left]),
            susceptibility=random.uniform(0.05, 0.55),
            ranker='unranked'
        ) for i in range(n_users)
    ]


def generate_documents(n_docs: int = 30) -> list[Document]:
    documents = [
        Document(
            name=f'Doc{str(i).zfill(2)}',
            lean=np.clip(random.normal(0, 0.4), -1, 1),
            relevance=0,
            clicks=0,
            exposure=0,
            pred_relevance=1,
            group=0
        ) for i in range(n_docs)
    ]
    for document in documents:
        document.group = np.sign(document.lean)
    return documents


def generate_qrels(users: list[User], documents: list[Document]) -> dict[(User, Document), int]:
    df = pd.DataFrame(users).merge(pd.DataFrame(documents),
                                   how='cross', suffixes=['_user', '_doc'])
    df['prob_relevance'] = np.exp(
        (-(df['lean_user']-df['lean_doc'])**2)/(2*df['susceptibility']**2))
    df['relevance'] = random.binomial(n=1, p=df['prob_relevance'])
    qrels = df.set_index(['name_user', 'name_doc'])['relevance'].to_dict()
    return qrels


class Ranker():
    def __init__(
        self, documents: list[Document], qrels: dict[(User, Document), int],
        seed: int, run_name: str='test', 
        base_system: str='baseline', fairco: bool=False, 
        fairness_condition: str='pred-relevance', 
        n_manip: int=0, queue_length: int=100,
        block = True, labda = 0.5,
        **kwargs
    ) -> None:
        self.documents = documents
        self.qrels = qrels
        self.run_name = f'{seed}_{run_name}'
        self.functions = {
            'baseline': baseline_ranker,
            'oracle': oracle,
            'worst_case': worst_case,
            'random': random_ranker,
            'clicks': naive_ranker,
        }
        self.base_ranker = base_system
        self.fairco = fairco
        self.condition = fairness_condition
        self.manipulation = int(n_manip)
        self.ranked = {key: 0 for key in self.functions}
        self.run_list = []
        self.right_threshold = 0.45
        self.left_threshold = 0.15
        self.queue_length = queue_length  # settable experimentally, but 100 worked well for now
        self.block = block
        self.labda = labda

    def update_unfair(self, user_index: int, exposure_balance : float) -> None:
        '''Updates the variables manipulation uses to function'''
        if self.block: 
            check_value = exposure_balance*user_index
            if check_value > 50:
                self.left_threshold = 0.5
                self.right_threshold = 0.54
            elif check_value > 30:
                self.left_threshold = 0.25
                self.right_threshold = 0.53
            elif check_value > 10:
                self.left_threshold = 0.15
                self.right_threshold = 0.52
            elif check_value < -10:
                self.left_threshold = 0.08
                self.right_threshold = 0.4
            elif check_value < -30:
                self.right_threshold = 0.35
                self.left_threshold = 0.07
            elif check_value < -50:
                self.right_threshold = 0.15 
                self.left_threshold = 0.06
            else:
                self.left_threshold = 0.08
                self.right_threshold = 0.52
        else:      
            self.right_threshold += exposure_balance * self.labda / 100
            self.left_threshold += exposure_balance * self.labda / 100 


    def exposure_balance(self) -> float:
        if self.condition == 'exposure':
            #TODO: Consider dividing exposure_balance by number of runs to make sure it's in roughly the same domain
            exposure_balance = np.mean([document.exposure for document in self.documents if document.group == 1]) - np.mean(
                [document.exposure for document in self.documents if document.group == -1])
        elif self.condition == 'pred-relevance':
            exposure_balance = (
                np.mean([document.exposure for document in self.documents if document.group == 1]) / np.mean([document.pred_relevance for document in self.documents if document.group == 1]) -
                np.mean([document.exposure for document in self.documents if document.group == -1]) / np.mean([document.pred_relevance for document in self.documents if document.group == -1])
            )
        elif self.condition == 'relevance':
            exposure_balance = (
                np.mean([document.exposure for document in self.documents if document.group == 1]) / np.mean([document.relevance for document in self.documents if document.group == 1]) -
                np.mean([document.exposure for document in self.documents if document.group == -1]) / np.mean([document.relevance for document in self.documents if document.group == -1])
            )
        elif self.condition == 'clicks':
            exposure_balance = (
                np.mean([document.clicks for document in self.documents if document.group == 1]) / np.mean([document.pred_relevance for document in self.documents if document.group == 1]) -
                np.mean([document.clicks for document in self.documents if document.group == -1]) / np.mean([document.pred_relevance for document in self.documents if document.group == -1])
            )

        else:
            raise KeyError(f'Unknown condition: {self.condition}')
        return exposure_balance

    def rank(self, user: User, i : int) -> None:
        exposure_balance = self.exposure_balance()
        if self.manipulation:
            self.update_unfair(user_index=i, exposure_balance=exposure_balance)
            if user.susceptibility > min(0.54, self.right_threshold):
                direction = 1 
                skew = 'right'
            elif user.susceptibility < max(0.06, self.left_threshold):
                skew = 'left'
                direction = -1
            else:
                skew = 'baseline'
                direction = 0

        for document in self.documents:
            document.rank_helper = self.functions[self.base_ranker](document)

            if self.fairco:
                document.rank_helper -= .01 * i * i * exposure_balance * document.group
                #TODO: Replace the .01 with a lambda 
                #TODO: Test this, this is kind of scary
        
        if self.manipulation:
            self.documents.sort(key= lambda x: (x.group * direction, x.rank_helper), reverse=True)
            for document in self.documents[:self.manipulation]:
                if document.group == direction:
                    document.rank_helper += i
                else:
                    continue
        self.documents.sort(key=lambda x: x.rank_helper, reverse=True)
        self.ranked[self.base_ranker] += 1
        user.ranker = skew if self.manipulation else self.base_ranker
        #print(f'{i} \t {exposure_balance} \t {self.right_threshold} \t {self.left_threshold} \t {user.ranker}')

        for rank, document in enumerate(self.documents):
            user_relevance = self.qrels[user.name, document.name]
            user_exposure = 1/np.log2(rank + 2)
            user_seen = random.binomial(1, user_exposure)
            user_clicked = user_relevance * user_seen
            self.run_list.append([user.name, 'Q0', document.name, rank, document.pred_relevance, user.ranker, document.rank_helper, user_clicked])
            #fairness_list.append([user.name, user.lean, document.name, document.lean, user_relevance, user_exposure, document.relevance, document.pred_relevance, rank])
            document.exposure += user_exposure
            document.clicks += user_clicked
            document.relevance += user_relevance
            document.pred_relevance += user_clicked / user_exposure
            #user.query_lean += document.group * user_exposure / group_size[document.group]

    def save(self, qrels, users: list[User], experiment_name: str = 'test', ) -> None:
        if not os.path.exists(f'runs/{experiment_name}'):
            os.mkdir(f'runs/{experiment_name}')

        with open(f'runs/{experiment_name}/qrels_{self.run_name}.txt', 'w') as f:
            for (user, doc), rel in qrels.items():
                f.write(f'{user} 0 {doc} {rel}\n')

        pd.DataFrame(self.run_list).to_csv(
            f'runs/{experiment_name}/{self.run_name}.run', sep=' ', header=False, index=False)
        pd.DataFrame(self.documents)[['name', 'lean', 'group']].to_csv(
            f'runs/{experiment_name}/groups_{self.run_name}.txt', sep=' ', header=False, index=False)
        pd.DataFrame(users)[['name', 'lean', 'susceptibility', 'ranker']].to_csv(
            f'runs/{experiment_name}/users_{self.run_name}.txt', sep=' ', header=False, index=False)


# %%
def main(file_name):
    seed_dict = {'base_experiment.xlsx' : 111111, 'discrete_long.xlsx' : 65432, 'labda_test.xlsx' : 55555, 'skew_pop.xlsx' : 123123123}
    configs = pd.read_excel(file_name).to_dict(orient='index')
    for config in configs.values():
        print(config)
        for seed in range(seed_dict[file_name], seed_dict[file_name] + 30): 
            random.seed(seed)
            user_list = generate_users(float(config['left_proportion']), int(config['n_users']))
            #experiment_name = f'{100*left_proportion:.0f}left'
            experiment_name = file_name.split('.')[0]
            document_list = generate_documents(int(config['n_docs']))
            qrels_dict = generate_qrels(user_list, document_list)
            ranker = Ranker(
                document_list,
                qrels_dict,
                seed,
                **config)
            for i, user in enumerate(user_list):
                ranker.rank(user, i)
            ranker.save(qrels_dict, user_list, experiment_name)




#%%
experiment = 'base_experiment.xlsx'
main(experiment) #This line is the only one that has to be changed to reproduce experimental results
#Fill in the name of the file containing the experiment you want to run, and it should produce the graph as in the paper
#EXCEPTION: For Figure 5, instead, run first this notebook, then fair_trec_eval.py  
if experiment != 'skew_pop.xlsx':
    graph(experiment.split('.')[0])
# %%
