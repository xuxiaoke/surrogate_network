'''
Estimation of Degree-Dependent Average Clustering  (CCK)
and Joint Degree Distribution (JDD) using random walk star samples

More details in:
M. Gjoka, M. Kurant, A. Markopoulou
2.5K-Graphs: from Sampling to Generation,
under submission
'''

__author__ = """Minas Gjoka"""

import sys, copy, math,random

import networkx as nx
import scipy as sci
import numpy as np
from scipy.io import loadmat
from scipy import ndimage

from sampling3 import *

def doubledict_sum(h):
    return sum(h[i][j] for i in h for j in h[i] )

def degree_dist(nkk):
    return dict( [ (d, sum(nkk[d].values())/float(d) ) for d in nkk] )

def countOccur(input_list):
    h = {}
    for elem in input_list:
        h[elem] = h.get(elem,0) + 1
    return h


def convert(h, fromtype='doubledict', totype='matrix'):      
    if fromtype=='doubledict':
        if totype == 'matrix':
            max_degree = max(h)
            
            adjm = np.zeros(  (max_degree+1, max_degree+1) )
            for ki in h.keys():
                for kj in h[ki]:
                    adjm[ki,kj] = h[ki][kj]
            
            return adjm
    elif fromtype=='singledict':
        if totype=='matrix':
            max_degree = max(h) 
            
            adjm = np.zeros(  (max_degree+1, max_degree+1) )
            for ki, kj in h.keys():
                adjm[ki,kj] = h[ki,kj]
            
            return adjm
    
    elif fromtype=='matrix':
        output = {}
        if totype =='doubledict':
            for k1 in range(len(h)):
                if sum(h[k1]>0) >0 :
                    output[k1] = {}
                    for k2 in range(len(h[k1])):
                        if h[k1][k2]>0:
                            output[k1][k2] = h[k1][k2]
        return output

def getinstance_stochastic(jdd):
    output = {}

    for ki in jdd:
        output[ki] = {}
        
    for ki in jdd:
        for kj in jdd[ki]:
            if ki<kj:
                decimal_part = jdd[ki][kj] - math.floor(jdd[ki][kj])            
                integer_part = math.floor(jdd[ki][kj])   
                if random.random() > decimal_part:
                    if integer_part>0:
                        output[ki][kj] = integer_part
                        output[kj][ki] = integer_part
                else:
                    output[ki][kj] = integer_part + 1
                    output[kj][ki] = integer_part + 1
            elif ki==kj:
                
                decimal_part = jdd[ki][kj] - math.floor(jdd[ki][kj])            
                integer_part = math.floor(jdd[ki][kj])   
                if random.random() > decimal_part:
                    if integer_part - decimal_part>0:
                        output[ki][kj] = integer_part - int(integer_part % 2 )
                else:
                    output[ki][kj] = integer_part + 1 + int(not integer_part % 2 )
                                
    return output
    
def is_realizable(jdd,check_dd=False,verbose=False):    
    dd = {}
    for row in jdd:
        rowsum  = sum([jdd[row][col] for col in jdd[row]])
        dd[row] = max(1,round( rowsum/float(row) ))

    
    violations = []
    for ki in jdd:
        for kj in jdd[ki]:
            if (ki == kj) and jdd[ki][kj] > (dd[ki]*(dd[ki]-1)):
                violations.append( (ki,kj, jdd[ki][kj], dd[ki]*(dd[ki]-1)) )
            elif (ki == kj) and jdd[ki][kj] % 2 == 1:
                violations.append( (ki,kj, jdd[ki][kj], jdd[ki][kj] - 1  ) )                
            elif (ki != kj) and jdd[ki][kj]>(dd[ki]*dd[kj]): 
                violations.append( (ki,kj, jdd[ki][kj],dd[ki]*dd[kj]) )
    if verbose:
        print "Nodes:%d  Edges:%d" % (sum(dd.values()), doubledict_sum(jdd))
    
    if check_dd:
        for row in sorted(jdd):
            rowsum  = sum([jdd[row][col] for col in jdd[row]])
            if rowsum/float(row) % round(rowsum/float(row)) != 0 :
                print row, rowsum/float(row)
                
    return violations


def smooth_matrix(input_kk, sigma, is_contract=1):
    ## input_kk is doubledict

    
    M = convert(input_kk,fromtype='doubledict', totype='matrix')
    if is_contract:
        rows = [i for i in range(len(M)) if sum(M[:,i])>0]
        M_contracted = M[rows,:][:,rows]        
        M_contracted = sci.ndimage.gaussian_filter(M_contracted,sigma)
        
        M_expanded = np.zeros( (len(M),len(M)))
        M_expanded[np.ix_(rows,rows) ] = M_contracted
        
        H = convert(M_expanded,fromtype='matrix', totype='doubledict')
    else:
        H = convert(sci.ndimage.gaussian_filter(M,sigma),fromtype='matrix', totype='doubledict')
        for i in H.keys():
            if i==0:
                del H[0]
            else:
                for j in H[i].keys():
                    if j==0:
                        del H[i][0]

    
                
    return H


class Estimation:
    
    def __init__(self,fname=None):        
        self.JDD = {}
        self.K_TRI = {}
        self.KAVG_TRI = {}
        self.CCK = {}
        self.g = None
        
        self.margin = 50
                            
    
    def load_graph(self, fname):
        print "Loading %s.." % (fname) ,
        sys.stdout.flush()
    
        if fname[-4:] == '.mat':        
            self.g = nx.from_scipy_sparse_matrix(loadmat(fname)['A'])
            print "Done"
        elif fname[-8:] == 'edges.gz':                    
            self.g = nx.read_edgelist(fname)        
            print "Done"
        else:
            print "Input not recognized. Supported filetypes: '.mat' and 'edges.gz'" 
            return

        self.JDD = {}
        self.K_TRI = {}
        self.KAVG_TRI = {}        
        self.CCK = {}
        
        self.n_edges = nx.number_of_edges(self.g)
        self.n_nodes = nx.number_of_nodes(self.g)
        self.avgdeg = round( 2.* self.n_edges / self.n_nodes)
        
        
    def sample(self, sample_type, p_sample):
        '''
         allowed sample types = ['rw','uis','wis']
        '''
        if not self.g:
            print "No graph loaded."
            return
        else:
            print "%.2f%% %s sample " % (p_sample*100, sample_type)
        
        self.JDD = {}
        self.K_TRI = {}
        self.KAVG_TRI = {}        
        self.CCK = {}
        
        self.sample_type = sample_type        
        self.sample_size = int(np.ceil(p_sample*nx.number_of_nodes(self.g)))       
        
        
        if sample_type=='uis':
            self.sample_list = list(uniform_independent_node_sample(self.g, size=self.sample_size))   
            self.hsample_weight = dict( (v,1.) for v in set(self.sample_list) )
        elif sample_type=='wis':
            self.sample_list = list(degree_weighted_independent_node_sample(self.g, size=self.sample_size))     
            self.hsample_weight = dict( (v, float(self.g.degree(v))) for v in set(self.sample_list))
        elif sample_type=='rw':            
            self.sample_list  = list(random_walk(self.g, size=self.sample_size))  
            self.hsample_weight = dict( (v, float(self.g.degree(v))) for v in set(self.sample_list)) 
            self.edge_sample_list =  [ (self.sample_list[i],v) for i,v in enumerate(self.sample_list[1:])]
                                    
            
    
    def estimate_JDD(self):
        '''
        input: sample_list, degrees of sampled nodes, number of edges
        output: JDD
        '''
        
        if 'estimate' in self.JDD:
            return  ## already estimated
        else:
            print "\nJDD Estimation  (110 dots)"
        
        nkk = {}
        norm_nkk = {}            
        
        if self.sample_type in ['rw']:  
            
            ##########Traversed Edges###########
            nkk['edge'] = {}
            norm_nkk['edge'] = 0

            count = 0
            dot_freq = int( math.ceil(.10*len(self.edge_sample_list)) ) 
            
            for (v,w) in self.edge_sample_list:                
                kv = self.g.degree(v)
                kw = self.g.degree(w)                
                
                nkk['edge'].setdefault(kv,{})[kw] = nkk['edge'].setdefault(kv,{}).get(kw,0) + 1.                        
                nkk['edge'].setdefault(kw,{})[kv] = nkk['edge'].setdefault(kw,{}).get(kv,0) + 1.                    
                norm_nkk['edge'] += 1
                
                ## progress bar
                count += 1
                if count % dot_freq == 0:
                    sys.stdout.write(".")
                    sys.stdout.flush()    
                ##  

            ##########Induced Edges###########
            nkk['ind'] = {}
            norm_nkk['ind'] = 0
            
            count = 0
            dot_freq = int( math.ceil(.01* sum([self.g.degree(v) for v in self.sample_list]) ) )
            
            hnode_count = countOccur(self.sample_list[self.margin+1:])            
            for idx,node in enumerate(self.sample_list):
                node_deg = self.g.degree(node)         
                                                                                                    
                for neighb in self.g[node]: 
                    ## progress bar
                    count += 1
                    if count % dot_freq == 0:
                        sys.stdout.write(".")
                        sys.stdout.flush()    
                    ##  
                    if (neighb in hnode_count):                
                        neighb_deg = self.g.degree(neighb)                            
                        
                        val = float(hnode_count[neighb]) / (self.hsample_weight[node]*self.hsample_weight[neighb])
                
                        nkk['ind'].setdefault(node_deg,{})[neighb_deg] = \
                           nkk['ind'].setdefault(node_deg,{}).get(neighb_deg,0) + val
                        
                        nkk['ind'].setdefault(neighb_deg,{})[node_deg] = \
                           nkk['ind'].setdefault(neighb_deg,{}).get(node_deg,0) + val
                                                
                        norm_nkk['ind'] += val
                        
                        
                ######
                ### add to left  side
                if idx-self.margin>=0:
                    node_added = self.sample_list[idx-self.margin]
                    hnode_count[node_added] = hnode_count.get(node_added,0) + 1
                
                ## remove from right side
                if idx+self.margin+1<self.sample_size:
                    node_removed = self.sample_list[idx+self.margin+1]
                    hnode_count[node_removed] = hnode_count.get(node_removed,0) - 1
                    if hnode_count[node_removed] == 0:
                        del hnode_count[node_removed]
                        
            
            ##########Normalize###########            
            for version in nkk:
                for kv in nkk[version]:
                    for kw in nkk[version][kv]:
                        nkk[version][kv][kw] /= norm_nkk[version]/float(self.n_edges)
                        
            ##########Hybrid###########
            threshold = self.avgdeg
            nkk['merged'] = {}
            for kv in nkk['edge']:
                for kw in nkk['edge'][kv]:
                    if (kv+kw)<threshold :
                        nkk['merged'].setdefault(kv,{})[kw] = nkk['edge'][kv][kw]    
                        
            for kv in nkk['ind']:
                for kw in nkk['ind'][kv]:
                    if (kv+kw)>=threshold :
                        nkk['merged'].setdefault(kv,{})[kw] = nkk['ind'][kv][kw]
                        
            
            ##########################
                        
            self.JDD['estimate'] = nkk['merged']
            
    def _realize_JDD_instance(self, sigma=2, verbose=False):
        
        if 'estimate' not in self.JDD:
            print "JDD not estimated yet"
            return        
        
        if 'realizable' in self.JDD:
            return self.changes        
        

        ##############Smoothing####################
        #############################################
        H = smooth_matrix(self.JDD['estimate'], sigma)
        jkk = getinstance_stochastic(H)
        degree_list = jkk.keys()
        
        
        
        ##############Realization####################
        #############################################
        ####
        #### dd = degree distribution
        ####
        dd = {}
        for row in jkk:
            rowsum  = sum([jkk[row][col] for col in jkk[row]])
            dd[row] = max(1,round( rowsum/float(row) ))
        
        ####
        #### remove realizability violations 
        #### (at this point diagonals are assumed to be even)
        ####
        violations = is_realizable(jkk)
        for (kv,kw, value, possible) in violations:
            if value> possible:
                jkk[kv][kw] = possible
                if jkk[kv][kw] == 0:
                    del jkk[kv][kw]
                    
        
        ### doubledict "out" will  contain final realizable 2K
        out = copy.deepcopy(jkk)
        
        n_total_changes = 0
        add_change = 0
        for row in sorted(degree_list,reverse=1):
            if row ==1:
                continue
            
            col_list = [col for col in jkk[row] if col<=row]
            rowsum  = sum([jkk[row][col] for col in col_list])  # 1 to row
            rowsum += sum([out[row][col]  for col in out.get(row,{}) if col>row] )  #row to n
            
                   
            rowsumtarget = row*dd[row]
                
                
            val_change = rowsumtarget - rowsum
            n_total_changes += abs(val_change)
            if verbose:
                print row, val_change
            if val_change == 0: ## do nothing
                pass   
            
            elif val_change >0:  ## add edges
                if len(col_list)>0: 
                    candidate_list = col_list
                else: 
                    candidate_list = [col for col in degree_list if col<=row] 
                
                nchanges = val_change
                steps_nochange_switch, steps_nochange_fail  = 10*nchanges, 100*nchanges
                while val_change>0:
                    col = random.choice(candidate_list)
                    if (row!=col) and (out[row].get(col,0)<dd[row]*dd[col]):
                        out[row][col] = out[row].get(col,0) + 1
                        out[col][row] = out[col].get(row,0) + 1
                        val_change -= 1
                        steps_nochange_switch, steps_nochange_fail  = 10*nchanges, 100*nchanges
                    elif (row==col) and (out[row].get(col,0)< (dd[row]*(dd[row]-1))/2 ) and (val_change>=2):
                        out[row][col] = out[row].get(col,0) + 2
                        val_change -= 2
                        steps_nochange_switch, steps_nochange_fail  = 10*nchanges, 100*nchanges
                    else:
                        steps_nochange_switch -= 1
                        steps_nochange_fail -= 1
                        
                        
                    if steps_nochange_switch<0:
                        candidate_list = [col for col in degree_list if col<=row] 
                    if steps_nochange_fail<0:
                        if verbose:
                            print "Attempt failed at degree %d. Try again with another stochastic instance." % (row)
                        return -1
                    
            elif val_change <0: 
                if sum([jkk[row].get(col,0) for col in col_list]) >= abs(val_change): ## remove edges
                    candidate_list = copy.deepcopy(col_list)
                    nchanges = abs(val_change)
                    steps_nochange_fail = 100*nchanges
                    while val_change <0:
                        col = random.choice(candidate_list)                        
                        if out[row][col]>0: ## edges available
                            if (row!=col) and (out[row].get(col,0)-1 <=dd[row]*dd[col]):
                                out[row][col] = out[row].get(col,0) - 1
                                out[col][row] = out[col].get(row,0) - 1                     
                                val_change += 1
                                steps_nochange_fail = 100*nchanges
                            elif (row==col) and (out[row].get(col,0)-2<=(dd[row]*(dd[row]-1))/2 ) and (val_change<=-2):
                                out[row][col] = out[row].get(col,0) - 2                                         
                                val_change += 2
                                steps_nochange_fail = 100*nchanges
                            else:
                                steps_nochange_fail -= 1
                                
                        else: ##edges NOT available. delete.
                            candidate_list.remove(col)
                            
                        if steps_nochange_fail<0:
                            if verbose:
                                print "Attempt failed at degree %d. Try again with another stochastic instance." % (row)
                            return -1
                        
                else:  ## add_edges
                    add_change = row + (val_change % (-row)  )
                    
                    
                    if len(col_list)>0: 
                        candidate_list = col_list
                    else: 
                        candidate_list = [col for col in degree_list if col<=row] 
                    
                    nchanges = add_change
                    steps_nochange_switch, steps_nochange_fail  = 10*nchanges, 100*nchanges
                    while add_change>0:
                        col = random.choice(candidate_list)
                        if (row!=col) and (out[row].get(col,0)<dd[row]*dd[col]):                            
                            out[row][col] = out[row].get(col,0) + 1
                            out[col][row] = out[col].get(row,0) + 1
                            add_change -= 1
                            steps_nochange_switch, steps_nochange_fail  = 10*nchanges, 100*nchanges
                        elif (row==col) and (out[row].get(col,0)< (dd[row]*(dd[row]-1))/2 ) and (add_change>=2):
                            out[row][col] = out[row].get(col,0) + 2
                            add_change -= 2  
                            steps_nochange_switch, steps_nochange_fail  = 10*nchanges, 100*nchanges
                        else:
                            steps_nochange_switch -= 1
                            steps_nochange_fail -= 1
                            
                            
                        if steps_nochange_switch<0:
                            candidate_list = [col for col in degree_list if col<=row] 
                        if steps_nochange_fail<0:
                            if verbose:
                                print "Attempt failed at degree %d. Try again with another stochastic instance." % (row)
                            return -1
        
        self.pcnt_changes = float(n_total_changes)/doubledict_sum(out)
        violations = is_realizable(out,check_dd=1)
        success = not len(violations)

        if verbose:
            print "\nAdded/removed %d edges (%.3f)\n\n" % (n_total_changes, pcnt_changes)                                
            print "Success: %d" % ( success)
            

        if success:
            self.JDD['realizable'] = out
        else:
            self.pcnt_changes = -1
        
        return self.pcnt_changes
        
    
    def realize_JDD(self, verbose=False):
        done = False
        attempts = 20
        
        if 'realizable' in self.JDD:
            return True
        else:
            print "\nPostprocessing to make JDD realizable"
                    
        while (attempts>0) and (not done):
            sys.stdout.write('.')
            sys.stdout.flush()
            attempts -= 1
            result = self._realize_JDD_instance(verbose)
            if not (result<0):
                done = True
                
        return done
        
        
    def estimate_KAVG_TRI(self):
        '''
        input: sample_list, degrees of sampled nodes
        output: CCK
        '''
        
        if 'estimate' in self.KAVG_TRI:
            return # already estimated
        else:
            print "\nCCK estimation  (110 dots)"
        
        cck = {}
        norm_cck = {}
                
        if self.sample_type in ['rw']:  

            ##########Traversed Edges###########
            cck['edge'] =  {}
            norm_cck['edge'] = {}
                
            count = 0
            dot_freq = int( math.ceil(.10*len(self.edge_sample_list)) ) 
            
            for (v,w) in self.edge_sample_list:                
                kv = self.g.degree(v)
                kw = self.g.degree(w)

                n_sharedpartners = float( len( set(self.g[v]) & set(self.g[w]) ) )
                
                cck['edge'][kv] = cck['edge'].get(kv,0) + n_sharedpartners                
                cck['edge'][kw] = cck['edge'].get(kw,0) + n_sharedpartners
                
                norm_cck['edge'][kv] = norm_cck['edge'].get(kv,0) + 1.
                norm_cck['edge'][kw] = norm_cck['edge'].get(kw,0) + 1.
                                   
                ## progress bar
                count += 1
                if count % dot_freq == 0:
                    sys.stdout.write(".")
                    sys.stdout.flush()    
                ##                  

            ##########Induced Edges###########
            cck['ind'] = {}
            norm_cck['ind'] = {}
            
            count = 0
            dot_freq = int( math.ceil(.01* sum([self.g.degree(v) for v in self.sample_list]) ) )
            
            
            hnode_count = countOccur(self.sample_list[self.margin+1:])
            for idx,node in enumerate(self.sample_list):
                node_deg = self.g.degree(node)         
                                                                                        
                neighbor_set = set(self.g[node])                
                for neighb in neighbor_set:   
                    ## progress bar
                    count += 1
                    if count % dot_freq == 0:
                        sys.stdout.write(".")
                        sys.stdout.flush()    
                    ##  
                    
                    if (neighb in hnode_count):                
                        neighb_deg = self.g.degree(neighb)                            
                        n_sharedpartners = float( len( neighbor_set & set(self.g[neighb]) ) )
                        
                        val = float(hnode_count[neighb])/self.hsample_weight[neighb]
                        
                        cck['ind'][node_deg]   = cck['ind'].get(node_deg,0) + (n_sharedpartners*val)
                        norm_cck['ind'][node_deg] = norm_cck['ind'].get(node_deg,0) + val

                
                ######
                ### add to left  side
                if idx-self.margin>=0:
                    node_added = self.sample_list[idx-self.margin]
                    hnode_count[node_added] = hnode_count.get(node_added,0) + 1
                
                ## remove from right side
                if idx+self.margin+1<self.sample_size:
                    node_removed = self.sample_list[idx+self.margin+1]
                    hnode_count[node_removed] = hnode_count.get(node_removed,0) - 1
                    if hnode_count[node_removed] == 0:
                        del hnode_count[node_removed]
            
            ##########Normalize#################            
            for version in cck:
                for kv in  cck[version]:
                    cck[version][kv] /= (norm_cck[version][kv]/(float(kv)) )
            
            
            ##########Hybrid###########
            threshold = self.avgdeg
            cck['merged'] = {}
            for kv in cck['edge']:
                if kv<threshold:
                    cck['merged'][kv] = cck['edge'][kv]
                            
            for kv in cck['ind']:
                if kv>=threshold:
                    cck['merged'][kv] = cck['ind'][kv]       
                    
            ############################
            
            self.KAVG_TRI['estimate'] = cck['merged']
        
    def estimate_CCK(self):
        '''
        input: sample_list, degrees of sampled nodes
        output: CCK
        '''
        
        if 'estimate' not in self.KAVG_TRI:
            self.estimate_KAVG_TRI()
        
        if 'estimate' not in self.JDD:
            self.estimate_JDD()
            
        if 'realizable' not in self.JDD:
            done = self.realize_JDD()
            
        if 'realizable' in self.JDD:
            nk = degree_dist(self.JDD['realizable'])
            self.K_TRI['estimate'] = dict([ (k,avg_tri*nk[k])  for (k,avg_tri) in  self.KAVG_TRI['estimate'].items() if (k>1) and (k in nk)])
            self.CCK['estimate'] = dict([ (k,avg_tri/float(k*(k-1)))  for (k,avg_tri) in  self.KAVG_TRI['estimate'].items() if (k>1) and (k in nk)])
            
    
    def calcfull_JDD(self):
        
        if 'full' in self.JDD:
            return 
        else:
            print "\nJDD full calculation  (100 dots)"

        count = 0
        dot_freq = int( math.ceil(.01* nx.number_of_edges(self.g) ) )

        nkk = {}
        for node in self.g.nodes_iter():
            node_degree = self.g.degree(node)        
            for neighb in self.g[node]:                    
                if node<neighb:             
                    ## progress bar
                    count += 1
                    if count % dot_freq == 0:
                        sys.stdout.write(".")
                        sys.stdout.flush()    
                    ##        
                    
                    neighb_degree = self.g.degree(neighb) 
                    
                    nkk.setdefault(node_degree,{})[neighb_degree] = \
                       nkk.setdefault(node_degree,{}).get(neighb_degree,0) + 1.
                    
                    nkk.setdefault(neighb_degree,{})[node_degree] = \
                        nkk.setdefault(neighb_degree,{}).get(node_degree,0) + 1.
        
        self.JDD['full'] = nkk
                    
    def calcfull_KAVGTRI(self):
        self.calcfull_CCK()
        
    def calcfull_CCK(self):
        
        if 'full' in self.CCK:
            return
        else:
            print "\nCCK full calculation (100 dots)"
            
        count = 0
        dot_freq = int( math.ceil(.01* nx.number_of_edges(self.g) ) )
            
        
        cck = {}
        for node in self.g.nodes_iter():
            node_degree = self.g.degree(node)        
            neighbor_set = set(self.g[node])
            for neighb in neighbor_set:                                                  
                if node<neighb:                
                    ## progress bar
                    count += 1
                    if count % dot_freq == 0:
                        sys.stdout.write(".")
                        sys.stdout.flush()    
                    ##                         
                    neighb_degree = self.g.degree(neighb)                                                            
                    n_sharedpartners = float( len( neighbor_set & set(self.g[neighb])) )  
                    
                    cck[ node_degree ] = cck.get(node_degree,  0) + n_sharedpartners
                    cck[ neighb_degree ] = cck.get(neighb_degree, 0) + n_sharedpartners
        
        
        nk = dict(  (i,k) for i,k in enumerate( np.bincount( [ self.g.degree(v) for v in  self.g ] ) ) if k>0  )         

        
        self.K_TRI['full'] = cck                
        self.KAVG_TRI['full'] = dict([ (k,tri/nk[k])  for (k,tri) in  self.K_TRI['full'].items() if (k>1) and (k in nk)])
        self.CCK['full'] = dict([ (k,tri/(nk[k]*k*(k-1)))  for (k,tri) in  self.K_TRI['full'].items() if (k>1) and (k in nk)]) 

    
    def get_JDD(self, kind):
        return self.JDD[kind]
    
    def get_CCK(self, kind):
        ###  C(k) = KTRI/(Nk[k]*k*(k-1))
        ###
        return self.CCK[kind]

    def get_KAVGTRI(self, kind):
        ###  KAVGTRI(k) = KTRI/Nk[k]
        return self.KAVG_TRI[kind]
    
    def get_KTRI(self, kind):        
        return self.K_TRI[kind]
    
    
    
def test_module(fname):    
    
    myest = Estimation()
    myest.load_graph(fname)
    myest.sample('rw',0.4)

    print "---\n\n"
    print "Calculating Full JDD"
    myest.calcfull_JDD()    
    print "\nCalculating Full CCK"
    myest.calcfull_CCK()


    print "\nSuccessful calculation of JDD and CCK"
    print "You can access full JDD by using function  get_JDD('full')"
    print "You can access full CCK by using function  get_CCK('full')"
    print "\n\n---\n\n"

    print "Estimating JDD"
    myest.estimate_JDD()    
    print "\nEstimating CCK"   
    myest.estimate_CCK()   # also makes realizable JDD


    print "\nSuccessful estimation of JDD and CCK"
    print "You can access estimated JDD by using function  get_JDD('estimate')"
    print "You can access realizable JDD by using function get_JDD('realizable')\n"
    print "You can access estimated CCK by using function  get_CCK('estimate')"
    print "You can access estimated K_TRI by using function  get_KTRI('estimate')"
    print "\n\n"
    
    return myest
    
if __name__ == "__main__":
    
    print "Testing Estimation module"
    
    #fname = "Facebook-New-Orleans.edges.gz"
    fname = "Caltech36.mat"        
    myest = test_module(fname)
    