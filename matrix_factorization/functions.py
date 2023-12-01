
import numpy as np
import pandas as pd
import json
from numba import njit

import sqlalchemy as db

from time import time
from datetime import timedelta

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define a function to load the playlist table from the SQL database into the array R_list
def make_R_list_sql(conn, pid_limit=None, progress=5., fetch_size=10**6):
    if pid_limit is None:
        # Get number of pairings
        results = conn.execute(db.text("SELECT COUNT(*) FROM pairings"))
        N = results.fetchall()[0][0]
        
        # Get columns one by one
        results = conn.execute(db.text("SELECT pid, tid FROM pairings"))
    else:
        # Get number of pairings
        results = conn.execute(db.text(f"SELECT COUNT(*) FROM pairings WHERE pid<{pid_limit}"))
        N = results.fetchall()[0][0]
        
        # Get columns one by one
        results = conn.execute(db.text(f"SELECT pid, tid FROM pairings WHERE pid<{pid_limit}"))
    
    R_list = np.empty((N,2), dtype=int)
    
    i = 0
    time_0 = time()
    show_progress = False
    old_i = 0
    divisor = np.round(N*progress/100)
    while rows:=results.fetchmany(fetch_size):
        if progress is not None:
            # Show progress
            if i/divisor > old_i/divisor:
                time_t = time()
                print('{:.2f}%: {:.2f} sec'.format(i/N*100, time_t-time_0))
                
                time_0 = time_t
                old_i = i
        
        for row in rows:
            R_list[i,0] = int(row[0])
            R_list[i,1] = int(row[1])
            i+=1
    
    return R_list





# Create two dicts such that:
# - idx_to_elt gives the i-th element in the sorted list
# - elt_to_idx gives the position of elt in the sorted list
# We can afford to store this information rather than recomputing it every time
def list_to_dict(new_list):
    # We only care about unique elements
    # Also, np.unique returns the sorted list
    new_list = np.unique(new_list)
    
    elt_to_idx = {}
    idx_to_elt = {}
    
    idx=0
    for elt in new_list:
        idx_to_elt[idx] = elt
        elt_to_idx[elt] = idx
        
        idx+=1
    
    return elt_to_idx, idx_to_elt

# Replaces each entry of R_list with its indices as given by pid_to_idx and tid_to_idx
def array_to_idx(R_list, pid_to_idx, tid_to_idx):
    R_list_idx = np.empty( R_list.shape, dtype=int)
    
    for row in range(R_list.shape[0]):
        R_list_idx[row,0] = pid_to_idx[ R_list[row,0] ]
        R_list_idx[row,1] = tid_to_idx[ R_list[row,1] ]
    
    return R_list_idx

# Format an unseen R_list to be processed in our pipeline
# We can only process songs that are in tid_to_idx already, so we remove those we don't know
def format_new_R_list(R_list_new, tid_to_idx):
    # Remove tracks that we don't recognize
    tid_known = list(tid_to_idx.keys())
    R_list_new = R_list_new[ np.isin(R_list_new[:,1], tid_known), : ]
    
    pid_to_idx, idx_to_pid = list_to_dict(R_list_new[:,0])
    
    R_idx_new = array_to_idx(R_list_new, pid_to_idx, tid_to_idx)
    
    return R_list_new, R_idx_new, pid_to_idx, idx_to_pid






# Run an epoch of gradient descent where the iterations parameter is the number of iterations.
@njit
def run_epoch(R_idx_list, P, Q, alpha, llambda, iterations):
    oldP = P.copy()
    oldQ = Q.copy()
    f = np.shape(P)[1]
    
    for _ in range(iterations):
        newP = oldP.copy()
        newQ = oldQ.copy()
        for u,i in R_idx_list:
            dotprod = 0
            for feature in range(f):
                dotprod += oldP[u,feature] * oldQ[i,feature]
            error = dotprod - 1

            for feature in range(f):
                pf = oldP[u,feature]
                qf = oldQ[i,feature]
                newP[u,feature] -= alpha * (error * qf + llambda * pf) 
                newQ[i,feature] -= alpha * (error * pf + llambda * qf)
        oldP = newP
        oldQ = newQ
    return newP, newQ

@njit
def SqError(R_list, P, Q):
    result = 0
    
    # sum over R_list
    for pid, tid in R_list:
        result += (1 - P[pid,:]@Q[tid,:])**2
    
    return result

@njit
def MSE(R_list, P, Q):
    result = 0
    N = R_list.shape[0]
    
    # sum over R_list
    for pid, tid in R_list:
        result += (1 - P[pid,:]@Q[tid,:])**2
    
    return result/N

@njit
# MSE with L2 regularization
def error_function_l2(R_list, P , Q, llambda):
    result = 0
    
    #sum over R_list
    for row, col in R_list:
        result += (1 - P[row,:]@Q[col,:])**2
    
    result += llambda * (np.linalg.norm(P)**2 + np.linalg.norm(Q)**2)
    return result




def new_user_vec(Y, llambda):
    d,f = np.shape(Y)
    vec = np.linalg.inv(np.transpose(Y) @ Y + llambda * np.identity(f)) @ np.transpose(Y) @ np.ones((d,1))
    return np.transpose(vec)

def make_Pval(R_list_new, Q, llambda):
    _, f = np.shape(Q)
    
    new_pids = np.unique(R_list_new[:,0])
    P_val = np.zeros((len(new_pids), f))
    
    for pid in new_pids:
        # get list of tracks in the playlist
        tid_list = R_list_new[ R_list_new[:,0]==pid, 1]

        # x is the row of Pval corresponding to this pid
        x = new_user_vec(Q[tid_list,:], llambda)
        
        for feature in range(f):
            P_val[pid, feature] = x[0,feature]
        
    return P_val



# Use grid search to compute the validation cost for all combinations of values of f, alpha, and llambda
def grid_search(f_values, alpha_values, llambda_values, NUM_ITERATIONS, path='models'):
    nF = len(f_values)
    nA = len(alpha_values)
    nL = len(llambda_values)
    costs = np.zeros((nF,nA,nL))
    
    # Create folder to save trained models if it does not exist already
    pathlib.Path(path).mkdir(exist_ok=True)
    
    # Execute grid search
    num_file = 0
    time_start = time()
    for idf in range(nF):
        f = f_values[idf]
        for ida in range(nA):
            alpha = alpha_values[ida]
            for idl in range(nL):
                llambda = llambda_values[idl]

                time_0 = time()
                iter_count = nA*nL*idf + nL*ida + idl+1
                print(f'Processing {iter_count}/{nF*nA*nL}:')
                print(f'({idf+1},{ida+1},{idl+1})/({nF},{nA},{nL})')
                error = False

                # Initialize random values
                P_initial = np.random.normal(0, 0.1, (num_playlists, f))
                Q_initial = np.random.normal(0, 0.1, (num_songs, f))
                
                # Obtain P, Q with the chosen hyperparameters
                P_trained, Q_trained = run_epoch(R_idx_train, P=P_initial, Q=Q_initial, alpha=alpha, llambda=llambda, iterations=NUM_ITERATIONS)
                
                # Skip iteration if Q_trained contains a nan
                if np.isnan(Q_trained).any():
                    costs[idf,ida,idl] = np.nan
                    error = True
                else:
                    # Check results against the validation set
                    P_val = make_Pval(R_idx_val, Q_trained, llambda)
                    costs[idf,ida,idl] = MSE(R_idx_val, P_val, Q_trained)
                
                # Show progress
                time_t = time()
                print('Training:  ', end=' ')
                print(str( timedelta(seconds=time_t-time_0) )[:-4])
                
                time_0 = time()
                file_name = f'{path}/Q_trained_{num_file}'
                np.save(file_name, Q_trained)
                num_file += 1
                time_t = time()
                
                print('Saving:    ', end=' ')
                print(str( timedelta(seconds=time_t-time_0) )[:-4])
                
                print('Total time:', end=' ')
                print(str( timedelta(seconds=time_t-time_start) )[:-4])
                
                if error:
                    print('-- Failed -- found nan')
                print()
    
    return costs

# Plot the errors found by grid_search
def plot_grid_search(costs, f_values, alpha_values, llambda_values):
    nF = len(f_values)
    nA = len(alpha_values)
    nL = len(llambda_values)
    
    # Plot results
    for ida in range(nA):
        for idl in range(nL):
            alpha = '{:.1E}'.format(alpha_values[ida])
            llambda = '{:.1E}'.format(llambda_values[idl])

            cost_f = costs[:,ida,idl]
            plt.plot(f_values, cost_f, '.', label=(alpha,llambda))
            
            """
            idt = 0
            display = True
            while np.isnan(cost_f[idt]):
                idt += 1
                if idt == nL:
                    display = False
                    break
            if display:
                plt.text(f_values[0], cost_f[0], str((alpha,llambda)))
            """

    ax = plt.gca()
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.xlabel('Number of latent features')

    print('Additional parameters:')
    print('Alpha values:', alpha_values)
    print('llambda values:', llambda_values)

