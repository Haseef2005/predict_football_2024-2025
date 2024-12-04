import pandas as pd
import pickle

model = pickle.load(open('XGBoostmodel.pkl','rb'))
cols = ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt', 'xg', 'xga']
new_cols = [f'{c}_rollinig' for c in cols]

# i will predict ManUTD vs Arsenal at emirates stadium 4th December 2024
match_data_mutd = pd.DataFrame({
    'venue_Home': [0],
    'opp_code': [0],
    'hour': [20],
    'day_code': [2],
    'gf_rollinig': [((4 + 1 + 3)/3)],
    'ga_rollinig': [(0 + 1 + 0)/3],
    'sh_rollinig': [(11 + 11 + 13)/3],
    'sot_rollinig': [(5 + 4 + 3)/3],
    'dist_rollinig': [(10 + 10 + 10)/3],
    'fk_rollinig': [(0 + 1 + 0)/3],
    'pk_rollinig': [(0 + 0 + 0)/3],
    'pkatt_rollinig': [(0 + 0 + 0)/3],
    'xg_rollinig': [(1.1 + 0.8 + 0.8)/3],
    'xga_rollinig': [(0.6 + 1.6 + 0.6)/3]
})

match_data_arsenal = pd.DataFrame({
    'venue_Home': [1],
    'opp_code': [16],
    'hour': [20],
    'day_code': [2],
    'gf_rollinig': [(5 + 3 + 1)/3],
    'ga_rollinig': [(2 + 0 + 1)/3],
    'sh_rollinig': [(14 + 19 + 13)/3],
    'sot_rollinig': [(6 + 5 + 3)/3],
    'dist_rollinig': [(15.4 + 17.7 + 16.3)/3],
    'fk_rollinig': [(1 + 0 + 0)/3],
    'pk_rollinig': [(2 + 0 + 0)/3],
    'pkatt_rollinig': [(2 + 0 + 0)/3],
    'xg_rollinig': [(3.5 + 1.2 + 1.5)/3],
    'xga_rollinig': [(1.5 + 0.3 + 1.5)/3]
})

inverse_mapping = { 1: 'Win', 0: 'Not win'}
predicted_mutd_labels = inverse_mapping[model.predict(match_data_mutd)[0]]
predicted_ars_labels = inverse_mapping[model.predict(match_data_arsenal)[0]]

print(f'Manchester will: {predicted_mutd_labels} Arsenal')
print(f'Arsenal will: {predicted_ars_labels} Manchester United')