import pandas as pd

def check_form(team_name,matches_table,match_date):
    count = 0
    club_matches = matches_table[((matches_table['HomeTeam']== team_name )| (matches_table['AwayTeam']== team_name)) & (matches_table['Date'] < match_date)]
    last_5 = club_matches.tail(5)

    for i in range(len(last_5)):
        a = last_5.iloc[i]
        if a['HomeTeam'] == team_name:
            if a['FTHG'] > a['FTAG']:
                count+=3
            elif a['FTHG'] == a['FTAG']:
                count+=1
        else :
            if a['FTHG'] < a['FTAG']:
                count+=3.5
            elif a['FTHG'] == a['FTAG']:
                count+=1.5 
    return count


def calculate_stats(team_name, matches_table, match_date):
    club_matches = matches_table[((matches_table['HomeTeam']== team_name )| (matches_table['AwayTeam']== team_name)) & (matches_table['Date'] < match_date)]
    last_5 = club_matches.tail(5)
    stats = {
        'scored': 0, 'losed': 0, 
        'shots': 0, 'shots_target': 0, 'corners': 0 , 'efficiency' : 0 , 'clean_sheets' : 0 , 'market_prob' : 0
    }
    n = len(last_5)
    if n == 0 : return {k: 0 for k in stats}

    for i in range(n):
        q = last_5.iloc[i]
        if team_name == q['HomeTeam']:
            stats['scored'] += q['FTHG']
            stats['losed'] += q['FTAG']
            stats['shots'] += q['HS']
            stats['shots_target'] += q['HST']
            stats['corners'] += q['HC']
            if q['FTAG'] == 0: stats['clean_sheets'] += 1
            stats['market_prob'] += (1 / q['B365H'])
        else:
            stats['scored'] += q['FTAG']
            stats['losed'] += q['FTHG']
            stats['shots'] += q['AS']
            stats['shots_target'] += q['AST']
            stats['corners'] += q['AC']
            if q['FTHG'] == 0: stats['clean_sheets'] += 1
            stats['market_prob'] += (1 / q['B365A'])
    result = {k : v / n for k,v in stats.items()}
    if (result['shots_target']>0):
         result['efficiency'] = result['scored']/result['shots_target']
    else:
         result['efficiency'] = 0
    return result