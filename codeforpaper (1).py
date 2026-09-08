#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
import pandas as pd
import numpy as np
import os
import pandapower as pp
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
from collections import defaultdict


# In[5]:


samples_dir = r"..." #put your own directory here
json_files = ([f for f in os.listdir(samples_dir) if f.endswith('.json')])[:24] 
history = defaultdict(list) 
history_degrees = defaultdict(lambda: {'in': [], 'out': [], 'total': []})
role_hub_auth = defaultdict(lambda: defaultdict(lambda: {'hub': [], 'auth': []}))  


for hour_file in json_files: 
    net = pp.from_json(os.path.join(samples_dir, hour_file))

    #create the digraph of the net for each of the 24 hours
    G = nx.DiGraph() 
    G.add_nodes_from(sorted(net.bus.index)) 
    for idx, line in net.line.iterrows():
        f, t = int(line['from_bus']), int(line['to_bus']) 
        p_flow = net.res_line.loc[idx, 'p_from_mw'] 
        if p_flow >= 0: G.add_edge(f, t) 
        else: G.add_edge(t, f) 


        h_score, a_score = nx.hits(G, max_iter=100, normalized=True) 
    
    #create bipartite graph for each of the 24 hours to use the hopcroft karp algorithm
    nodi = list(G.nodes()) 
    H = nx.Graph() 
    nodi_l = [f"{n}_l" for n in nodi]
    nodi_r = [f"{n}_r" for n in nodi]
    H.add_nodes_from(nodi_l, bipartite=0) 
    H.add_nodes_from(nodi_r, bipartite=1) 
    for i, j in (G.edges()):  
        H.add_edge(f"{i}_l", f"{j}_r")

    matching = bipartite.hopcroft_karp_matching(H, top_nodes=set(nodi_l))

    #classify the nodes based on matching results
    for n in nodi:
        is_driver = f"{n}_r" not in matching 
        is_sensor = f"{n}_l" not in matching 

        if is_driver and is_sensor: ruolo = "Dual" 
        elif is_driver: ruolo = "Driver"
        elif is_sensor: ruolo = "Sensor"
        else: ruolo = "Ordinary" 
        
        history[n].append(ruolo) 
        history_degrees[n]['in'].append(G.in_degree(n)) 
        history_degrees[n]['out'].append(G.out_degree(n))
        history_degrees[n]['total'].append(G.degree(n))
        role_hub_auth[n][ruolo]['hub'].append(h_score[n])  
        role_hub_auth[n][ruolo]['auth'].append(a_score[n])


persistenza = []
for n, roles in history.items():
    total = len(roles) 
    counts = pd.Series(roles).value_counts(normalize=True) * 100 
    
    unique_roles = set(roles) 
    is_permanent = len(unique_roles) == 1
    main_role = roles[0] if is_permanent else "Dynamic"
    
    persistenza.append({
        'Node': n,
        'Driver': counts.get('Driver', 0),
        'Sensor': counts.get('Sensor', 0),
        'Dual': counts.get('Dual', 0),
        'Ordinary': counts.get('Ordinary', 0),
        'Permanent': is_permanent,
        'MainRole': roles[0]
    })

df_persist = pd.DataFrame(persistenza).sort_values(by='Node')

#table for permanent nodes
print("permanent nodes")
perm = df_persist[df_persist['Permanent']] 
for role in ["Driver", "Sensor", "Dual", "Ordinary"]:
    nodes = perm[perm['MainRole'] == role]['Node'].tolist() 
    print(f"{role}: {nodes}")

#table for dynamic nodes
print("dynamic nodes")
dinamici = round(df_persist[~df_persist['Permanent']],2)
print(dinamici[['Node', 'Driver', 'Sensor', 'Dual', 'Ordinary']].to_string(index=False))


# In[13]:


#mean
nodo_ruolo = dict(zip(df_persist['Node'], df_persist['MainRole'])) 

grado_nodi = []
for n in G.nodes():
    grado_nodi.append({
        'Node': n,
        'In_Degree':       np.mean(history_degrees[n]['in']),
         
        'Out_Degree':      np.mean(history_degrees[n]['out']),
        'Total_Degree':    np.mean(history_degrees[n]['total']),
        'Ds_total_Degree': np.std(history_degrees[n]['total']),
        'Role': nodo_ruolo[n]
    })
df_degrees = pd.DataFrame(grado_nodi) 

#mean degree given the node
avg_deg = df_degrees.groupby('Role')[['Total_Degree']].mean().sort_values('Total_Degree') 
print(avg_deg)

balance = df_degrees.groupby('Role')[['In_Degree', 'Out_Degree']].mean()
print(balance)

role_counts = df_degrees['Role'].value_counts()
print(role_counts)


# In[21]:


#calculate coefficient of variation and standard deviation
roles_list = ['Driver', 'Sensor', 'Dual', 'Ordinary']

grado_per_ruolo = {r: {'in': 0, 'out': 0, 'total': 0, 'n_nodi': 0} for r in roles_list}
scarti_quadratici = {r: {'in': 0, 'out': 0, 'total': 0} for r in roles_list}

for ora in range(24):
    for ruolo in roles_list:
        nodi_ruolo_ora = [n for n in G.nodes() if history[n][ora] == ruolo]
        
        if len(nodi_ruolo_ora) == 0:
            continue
        
        grado_per_ruolo[ruolo]['in']     += sum(history_degrees[n]['in'][ora]     for n in nodi_ruolo_ora)
        grado_per_ruolo[ruolo]['out']    += sum(history_degrees[n]['out'][ora]    for n in nodi_ruolo_ora)
        grado_per_ruolo[ruolo]['total'] += sum(history_degrees[n]['total'][ora] for n in nodi_ruolo_ora)
        grado_per_ruolo[ruolo]['n_nodi'] += len(nodi_ruolo_ora)

medie = {}
for ruolo in roles_list:
    n = grado_per_ruolo[ruolo]['n_nodi']
    medie[ruolo] = {
        'in':     grado_per_ruolo[ruolo]['in']     / n if n > 0 else 0,
        'out':    grado_per_ruolo[ruolo]['out']    / n if n > 0 else 0,
        'total': grado_per_ruolo[ruolo]['total'] / n if n > 0 else 0,
    }

for ora in range(24):
    for ruolo in roles_list:
        nodi_ruolo_ora = [n for n in G.nodes() if history[n][ora] == ruolo]
        
        if len(nodi_ruolo_ora) == 0:
            continue
        
        scarti_quadratici[ruolo]['in']     += sum((history_degrees[n]['in'][ora]     - medie[ruolo]['in'])**2     for n in nodi_ruolo_ora)
        scarti_quadratici[ruolo]['out']    += sum((history_degrees[n]['out'][ora]    - medie[ruolo]['out'])**2    for n in nodi_ruolo_ora)
        scarti_quadratici[ruolo]['total'] += sum((history_degrees[n]['total'][ora] - medie[ruolo]['total'])**2 for n in nodi_ruolo_ora)

risultati = []
for ruolo in roles_list:
    n = grado_per_ruolo[ruolo]['n_nodi']
    
    std_in     = np.sqrt(scarti_quadratici[ruolo]['in']     / (n-1)) if n > 0 else 0
    std_out    = np.sqrt(scarti_quadratici[ruolo]['out']    / (n-1)) if n > 0 else 0
    std_totale = np.sqrt(scarti_quadratici[ruolo]['total'] / (n-1)) if n > 0 else 0
    
    risultati.append({
        'Role':          ruolo,
        'In_Degree':      medie[ruolo]['in'],
        'Out_Degree':     medie[ruolo]['out'],
        'Total_Degree':   medie[ruolo]['total'],
        'Dev_Standard_Tot' : std_totale,
        'Dev_Standard_In': std_in,
        'Dev_Standard_Out' : std_out,
        'CV_In':          std_in     / medie[ruolo]['in']     if medie[ruolo]['in']     > 0 else 0,
        'CV_Out':         std_out    / medie[ruolo]['out']    if medie[ruolo]['out']    > 0 else 0,
        'CV_Total':       std_totale / medie[ruolo]['total'] if medie[ruolo]['total'] > 0 else 0,
    })

df_avg = pd.DataFrame(risultati).set_index('Role')
print(df_avg.sort_values('Total_Degree'))


# In[25]:


#plot to see how roles change for node 108
nodo = 108
ore = list(range(24))
in_deg = history_degrees[nodo]['in']
out_deg = history_degrees[nodo]['out']
ruoli = history[nodo]

plt.figure(figsize=(9, 3.5))
plt.plot(ore, in_deg, label='In-degree', marker='o', color='tab:blue')
plt.plot(ore, out_deg, label='Out-degree', marker='s', linestyle='--', color='tab:orange')


for ora in ore:
    plt.text(ora, 2.3, ruoli[ora][0], fontsize=8, ha='center', color='gray')

plt.title(f'Node {nodo} - In/Out degree over 24h')
plt.xlabel('Hour')
plt.ylabel('Degree')
plt.xticks(ore)
plt.yticks([0, 1, 2, 3])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'degree_nodo_{nodo}.pdf')
plt.show()

print(f"Ruoli nodo {nodo} nelle 24 ore:")
for ora, r in enumerate(ruoli):
    print(f"  Ora {ora:2d}: {r:10s} | in={in_deg[ora]}  out={out_deg[ora]}")


# In[22]:


#table (degree given node)
risultati2 = {'Driver': [], 'Sensor': [], 'Dual': [], 'Ordinary': []}

for nodo, roles in history.items():
    in_degs  = history_degrees[nodo]['in']
    out_degs = history_degrees[nodo]['out']
    
    for ora in range(24):
        ruolo   = roles[ora]
        in_deg  = in_degs[ora]
        out_deg = out_degs[ora]
        
        if in_deg > out_deg:
            condizione = 'in>out'
        elif out_deg > in_deg:
            condizione = 'out>in'
        else:
            condizione = 'in==out'
        
        risultati2[ruolo].append(condizione)

for ruolo, condizioni_lista in risultati2.items():
    totale = len(condizioni_lista)
    counts = pd.Series(condizioni_lista).value_counts(normalize=True) * 100
    print(f"\n{ruolo} ({totale} osservazioni):")
    for condizione in ['in>out', 'out>in', 'in==out']:
        print(f"  {condizione:10s}: {counts.get(condizione, 0):.1f}%")


# In[45]:


gen_rows = []

for hour_idx, hour_file in enumerate(json_files):
    net = pp.from_json(os.path.join(samples_dir, hour_file))
    bus_con_gen = set(net.gen['bus'].tolist() + net.ext_grid['bus'].tolist())
    nodi_ora = list(G.nodes())  
    ruoli_ora = {n: history[n][hour_idx] for n in nodi_ora}
    
    driver_ora = [n for n, r in ruoli_ora.items() if r == 'Driver']
    dual_ora   = [n for n, r in ruoli_ora.items() if r == 'Dual']
    
    gen_rows.append({
        'Ora':            hour_idx,
        'N_Driver':       len(driver_ora),
        'N_Dual':         len(dual_ora),
        'Driver_con_gen': sum(1 for n in driver_ora if n in bus_con_gen),
        'Dual_con_gen':   sum(1 for n in dual_ora   if n in bus_con_gen),
    })

df_gen = pd.DataFrame(gen_rows)


# In[51]:


df_gen['Tot_driver_dual']=df_gen['N_Driver'] + df_gen['N_Dual']
df_gen['Tot_con_gen']=df_gen['Driver_con_gen'] + df_gen['Dual_con_gen']

media=round(df_gen['Tot_con_gen'].sum() / df_gen['Tot_driver_dual'].sum() * 100, 2)

print(media)


# In[41]:


#hub and authority scores
results_hits = []
for n in history.keys():
    for ruolo, scores in role_hub_auth[n].items():
        results_hits.append({
            'Nodo': n,
            'Ruolo': ruolo,
            'Hub_Medio': np.mean(scores['hub']),
            'Auth_Medio': np.mean(scores['auth'])
        })

df_hits = pd.DataFrame(results_hits)
stats_hits = df_hits.groupby('Ruolo')[['Hub_Medio', 'Auth_Medio']].mean()
print(stats_hits.sort_values(by='Auth_Medio'))


# In[34]:


#plot to compare hub and authority scores 
baseline = 1 / 118
stats_hits['Hub_vs_baseline'] = stats_hits['Hub_Medio'] / baseline
stats_hits['Auth_vs_baseline'] = stats_hits['Auth_Medio'] / baseline

ax = stats_hits[['Hub_vs_baseline', 'Auth_vs_baseline']].plot(
    kind='bar',
    figsize=(7, 4),
    color=['tab:blue', 'tab:orange'],
    rot=0
)

plt.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Baseline')
plt.ylabel('Ratio to baseline (1/118)')
plt.xlabel('Role')
plt.title('Normalized Hub and Authority Scores')
plt.legend(['Hub Score', 'Authority Score', 'Baseline'])
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()


# In[13]:


#Mean hub and authority scores
hits_per_ruolo = {r: {'hub': 0, 'auth': 0, 'n_obs': 0} for r in ['Driver', 'Sensor', 'Dual', 'Ordinary']}

for n in history.keys():
    for ruolo, scores in role_hub_auth[n].items():
        hits_per_ruolo[ruolo]['hub']   += sum(scores['hub'])   
        hits_per_ruolo[ruolo]['auth']  += sum(scores['auth'])  
        hits_per_ruolo[ruolo]['n_obs'] += len(scores['hub'])  

# Divido alla fine
risultati_hits = []
for ruolo, vals in hits_per_ruolo.items():
    n = vals['n_obs']
    risultati_hits.append({
        'Ruolo':      ruolo,
        'Hub_Medio':  vals['hub']  / n if n > 0 else 0,
        'Auth_Medio': vals['auth'] / n if n > 0 else 0,
    })

df_hits = pd.DataFrame(risultati_hits).set_index('Ruolo')
print(df_hits.sort_values('Auth_Medio'))

