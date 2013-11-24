'''
Generation of 2.5K graphs given measured CCK and JDD.

More details in:
M. Gjoka, M. Kurant, A. Markopoulou
2.5K-Graphs: from Sampling to Generation,
under submission
'''

__author__ = """Minas Gjoka"""

import sys, copy, math, random, time, heapq
from collections import defaultdict
import itertools as it
import networkx as nx
import scipy as sci
import numpy as np

from Estimation import Estimation

def doubledict_sum(h):
    return sum(h[i][j] for i in h for j in h[i] )

def degree_dist(nkk):
    return dict( [ (d, sum(nkk[d].values())/float(d) ) for d in nkk] )

def countOccur(input_list):
    h = {}
    for elem in input_list:
        h[elem] = h.get(elem,0) + 1
    return h

def ktri_to_CCk(ktri, nk):
    return dict([ (k,float(ntri)/(nk[k]*k*(k-1)) ) for k,ntri in ktri.items() if (k>1) and (k in nk)])

def compute_diff_k(curr_k,target_k):
    diff_k = {}
    keys = set(curr_k.keys()).union( target_k.keys() )
    return dict([ (i,curr_k.get(i,0) - target_k.get(i,0)) for i in keys])

def singledict_nmae(g_k, h_k, sq=0):
    degree_list = set(g_k.keys()).union( h_k.keys() )    
    output = sum(   abs( h_k.get(i,0) - g_k.get(i,0) )    for i in degree_list  )         
    normal_factor =  sum(g_k.values())

    return (output/(1.*normal_factor))


def clustcoeff(k_tri, nk):
    return np.mean([float(sh)/(nk[k]*k*(k-1)) if k>1 else 0 for k,sh in k_tri.items() for j in range(int(nk[k]))  ])


class Stub:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        
        
class Lash:

    def __init__(self, inputList):
        ## inputList is a list of elements
        self.index_val = dict([ (index,val)  for index,val in enumerate(inputList)])
        self.val_index = dict([ (val,index)  for index,val in enumerate(inputList)])
        self.nextidx = len(self.index_val)  #  next available index


    def add(self,elem):
        ### assume that other operations never leave "gaps"
        if elem not in self.val_index: # does it exist?
            self.val_index[elem] = self.nextidx   # add as last
            self.index_val[self.nextidx] = elem
            self.nextidx += 1


    def delete(self,elem):
        ## find index associated with element
        index = self.val_index.get(elem, -1)

        if index != -1:

            if (index != self.nextidx-1): ## if elem is not the last element
                ## move last element to this index
                self.index_val[index] = self.index_val[self.nextidx-1]
                self.val_index[ self.index_val[self.nextidx-1] ] = index


            del self.index_val[self.nextidx -1]   ## delete last index
            del self.val_index[elem]              ## delete association of "elem"


            self.nextidx -= 1

    def get(self,index):
        return self.index_val[index]
    
    def get_all(self):
        return self.val_index.keys()

    def get_random(self):
        if self.nextidx>0:
            return random.choice(self.index_val)
        else:
            return []

    def get_randomlist(self, n):
        ### optimize
        #return [self.index_val[i] for i in random.sample(xrange(self.nextidx), min(n,self.nextidx) )]
        return random.sample( self.val_index, min(n,self.nextidx) )
    
    def size(self):
        return len(self.index_val)
        
        
        
class Generation:
    
    def __init__(self):
        self.JDD_target = None
        self.K_TRI_target = None
        self.Gdk = None  # 2K
        self.G = None  # 2.5K
        
        self.node_coord = None
        self.update_Freq = 10**3
    
    
    def set_JDD(self, input):
        self.JDD_target = input
        
    def set_KTRI(self, input):
        self.K_TRI_target = input



    def graph_properties(self,g):
        shared_vv = {}
        edges_deg= {}
        ktri = {}
        nkk = {}
        nk = {}    
        
        node_deg = dict( (v,g.degree(v)) for v in g.nodes_iter() )
        
        for kv in set(node_deg.values()):            
            shared_vv[kv] = {}        
            edges_deg[kv] = Lash([])
            ktri[kv] = 0
            nk[kv] = 0
    
        count = 0
        dot_freq = int( math.ceil(.01*nx.number_of_edges(g)) )
    
        sys.stdout.write(".")
        sys.stdout.flush()        
        for v1 in g.nodes_iter():        
            neighbors = set(g[v1])
            
            nk[node_deg[v1]] += 1.
            
            for v2 in neighbors:                        
                if v1<v2: ## assume undirected, do it once only
                    count += 1
                    if count % dot_freq == 0:
                        sys.stdout.write(".")
                        sys.stdout.flush()
                    
                    sharedpartners =  neighbors & set(g[v2])
                    n_sharedpartners = 1.* len( sharedpartners ) 
                            
                    v1_degree, v2_degree = node_deg[v1],node_deg[v2]
            
                    shared_vv.setdefault(v1,{})[v2] = shared_vv.setdefault(v2,{})[v1] = sharedpartners
            
                    ktri[v1_degree] += n_sharedpartners
                    ktri[v2_degree] += n_sharedpartners
                    
                    if (v1_degree>1) and (v2_degree>1):
                        edges_deg[v1_degree].add( (v1,v2) )
                        edges_deg[v2_degree].add( (v2,v1) )
    
        return shared_vv, ktri, edges_deg, nk

    
        
    def _get2k_random_degree(self, degree, nkk):
        ### input >  dict[k1, dict[k2, count]]
        ###
        if degree not in nkk:
            print "Error: degree %d not in nkk" % degree
    
        total_edgecount = sum( nkk[degree].values() )
        choice = random.random()*total_edgecount
    
        for k,count in nkk[degree].items():
            choice = choice - count
            if choice <=0 :
                return k
    
    
        return -1
    
    
    def _dkgen2k_connectstubs(self, stubs, randomstubs, degree_stub, available, nkk_working, G):
        notAvailFirst_Count = 0
        notAvailSec_Count = 0
        sameNode_Count = 0
        alreadyConnected_Count = 0
        targetDegree_Count = 0
        edge_Count = 0
          
            
        i = 0
        dot_freq = int( math.ceil(.01*len(randomstubs)) )
        while len(randomstubs)>0:
            i += 1
            if i % dot_freq ==0 :
                sys.stdout.write(".")
                sys.stdout.flush()
            
            stubid = randomstubs.pop()
    
            ## is stubid available?
            if not available[stubid]:
                notAvailFirst_Count += 1
                continue
    
            stub = stubs[stubid]
    
            # find degree of node that stub should be connected
            target_degree = self._get2k_random_degree(stub.degree, nkk_working)
            if target_degree<=0:
                targetDegree_Count += 1
                continue
    
            done = False
            stublist_iter = iter(degree_stub[target_degree])
            while (not done) and stublist_iter.__length_hint__()>0 :
    
                targetid = stublist_iter.next()
    
                ##is targetid available?
                if not available[targetid]:
                    notAvailSec_Count =+ 1
                    continue
    
                candidate_stub = stubs[targetid]
    
                ## stubs belong to same node?
                if stub.nodeid == candidate_stub.nodeid:
                    sameNode_Count += 1
                    continue
    
                ## nodes are already connected ?
                if G.has_edge(stub.nodeid, candidate_stub.nodeid) or \
                   G.has_edge(candidate_stub.nodeid, stub.nodeid):
                    alreadyConnected_Count += 1
                    continue
    
                G.add_edge(stub.nodeid, candidate_stub.nodeid)
                edge_Count += 1
    
                available[stubid] = 0
                available[targetid] = 0
    
                if nkk_working[stub.degree][candidate_stub.degree]:
                    nkk_working[stub.degree][candidate_stub.degree] -= 1
    
                if nkk_working[candidate_stub.degree][stub.degree]:
                    nkk_working[candidate_stub.degree][stub.degree] -= 1
    
                done = True
    
        

        
    def _vv_dist_key(self, (v,w) ):
        ## node_coord is a hash table with node coordinates
        
        d = abs(self.node_coord[v]-self.node_coord[w])
        return min( d, 1 - d  )    
        
    
    def _vv_dist(self, v,w ):
        ## node_coord is a hash table with node coordinates
        
        d = abs(self.node_coord[v]-self.node_coord[w])
        return min( d, 1 - d  )


    def _kneighb_pairs_batch(self,node_list, node_residual, kneighb, batch, node_borders):
        '''
        input:  
            node_list         
            node_residual
            kneighb
            batch = [0,1..]
            node_borders  = [ (left,right)]
            
                
        output:
            vv_list 
        
        
        '''
        vv_list = []    
        dot_freq =  int( math.ceil(0.10*len(node_list)) )
        n = len(node_list)
        
        wsize = min(n-1, kneighb*(batch+1) )
        for j,v in enumerate(node_list):
            
            if node_residual[v]:
                ## find k nearest neighbors
                left, right= node_borders[j]
                
                next_left = (left-1) % n
                next_right = (right+1) % n
                while ((right-left) % n )< wsize :
                    if self._vv_dist( v, node_list[ next_left ] ) <  self._vv_dist( v, node_list[ next_right]):
                        left = next_left
                        next_left = (left-1) % n
                        if node_residual[ node_list[ left] ]:
                            vv_list.append( (v, node_list[ left]) )                
                    else:
                        right = next_right
                        next_right = (right+1) % n
                        if node_residual[ node_list[ right] ]:
                            vv_list.append( (v, node_list[ right]) )
                node_borders[j] = (left,right)
                
                if j % dot_freq == 0:
                    sys.stdout.write("*")
                    sys.stdout.flush()
        return vv_list
    
        
        
    def _firstpass_allpair_smart(self, Gout, batch_size, jdd, h_degree_nodelist, node_residual, node_degree):
        
        print "\n-----------\n1st pass  allpairs smart"
        nEdges = doubledict_sum(jdd)/2
        nNodes = len(node_degree)
                     
    
        kneighb = min(nNodes, 2*math.ceil(batch_size/float(nNodes)))
        nBatches = int( nNodes / kneighb ) 
        print "Kneighb: %d\tnBatches: %d" % (kneighb, nBatches)
    
        node_list = node_degree.keys()
        node_list.sort( key=self.node_coord.__getitem__ )
        
        node_borders = [(i,i) for i in xrange(len(node_list))]
        
        print "Throwing edges"    
        pcntEdges_prev,  pcntNodes,  = 0.01, 0.01
        majorityDeg_prev = max([sum([node_residual[j] for j in h_degree_nodelist[i]]) for i in set(node_degree.values())] )
        
        edgecount = 0
        for batch in xrange(nBatches) :
            print batch, 
            vv_list = self._kneighb_pairs_batch( node_list, node_residual, kneighb, batch, node_borders )
            vv_list.sort( key = self._vv_dist_key )
                
            dot_freq = int( math.ceil(.10*len(vv_list)))
    
            for p_idx, (v,w) in enumerate(vv_list):
                if (node_residual[v]>0) and (node_residual[w]>0) \
                   and jdd.get(node_degree[v], {}).get( node_degree[w],0 ) > 0 \
                   and not Gout.has_edge(v,w) and v!=w:
                    
                    edgecount +=1
                    Gout.add_edge(v,w)
                    
                    node_residual[v] -= 1
                    node_residual[w] -= 1  
                        
                    
                    if node_degree[v] != node_degree[w]:
                        jdd[ node_degree[v] ][ node_degree[w] ] -= 1
                        jdd[ node_degree[w] ][ node_degree[v] ] -= 1
                    else:
                        jdd[ node_degree[v] ][ node_degree[w] ] -= 2
                    
                                        
        
                if p_idx % dot_freq == 0:
                    sys.stdout.write(".")
                    sys.stdout.flush()
            #endfor
            
            pcntEdges = edgecount/float(nEdges)
            pcntNodes = 1-len([1 for i in node_residual.values() if i>0])/ float(nNodes)
            majorityDeg = max([sum([node_residual[j] for j in h_degree_nodelist[i]]) for i in set(node_degree.values())] )
            print "%.3f, %.3f, %d\t%f" % ( pcntEdges, pcntNodes, majorityDeg, self._vv_dist_key(vv_list[-1] ) ) 
            if (pcntEdges-pcntEdges_prev)/(pcntEdges_prev) < 0.01 and \
               (pcntNodes-pcntNodes_prev)/(pcntNodes_prev) < 0.01 and \
               (majorityDeg_prev-majorityDeg)/float(majorityDeg_prev) < 0.02:
                break
            else:
                pcntEdges_prev = pcntEdges
                pcntNodes_prev = pcntNodes
                majorityDeg_prev = majorityDeg                    
        
                    
        print "\n[1st] allpairsmart residual stubs: %d/%d\n-----------\n" % ( sum( node_residual.values()),  doubledict_sum(jdd) ) 
        
        
    def _kneighb_pairs(self,alist,blist,k):
        na = len(alist)    
        nb = len(blist)
        
        if (na ==0) or (nb==0):
            return []
        
        vv_list = []
        idx= 0        
        
        for v in alist:
    
            d = self._vv_dist( v,blist[idx % nb] )
            ## find nearest neighbor
            
            dright = self._vv_dist( v,blist[ (idx+1) % nb] )
            dleft  = self._vv_dist( v,blist[ (idx-1) % nb] )
            if dright < dleft:            
                while dright < d:
                    d = dright
                    idx +=1                
                    dright = self._vv_dist( v,blist[ (idx+1) % nb] )
            else:
                while dleft < d:
                    d = dleft
                    idx -=1                
                    dleft = self._vv_dist( v,blist[ (idx-1) % nb] )
                
            vv_list.append( (v,blist[idx % nb]) )
            
            ## find k-1 nearest neighbors
            if nb>1:
                left= idx-1
                right= idx+1
                while abs(right-left) <=k :                
                    if self._vv_dist( v,blist[left % nb] ) <  self._vv_dist( v,blist[right % nb] ):
                        vv_list.append( (v,blist[left % nb]) )
                        left -= 1
                    else:
                        vv_list.append( (v,blist[right % nb]) )
                        right += 1
        return vv_list
        
    def _firstpass_kneighb(self, Gout, kneighb, jdd, h_degree_nodelist, node_residual):        
        
        print "\n-----------\n1st pass  kneighb"
        ##### order of visits
        kl_pairs = [ (i,j) for i in jdd for j in jdd[i] if jdd[i][j]>0 ]
        random.shuffle(kl_pairs)
        
        dot_freq = int( math.ceil(.01*len(kl_pairs)) )
        for p_idx, (k,l) in enumerate(kl_pairs):
            if k>=l:                                                        
                if k>l :
                    Jkl = jdd[k][l]
                    
                    k_unsat = [v for v in h_degree_nodelist[k] if node_residual[v]>0 ]
                    l_unsat = [w for w in h_degree_nodelist[l] if node_residual[w]>0 ]                                
                        
                    #### kneighb pairs only 
                    edge_list   =     self._kneighb_pairs(k_unsat, l_unsat, kneighb  )
                    edge_list.extend( ((j,i) for (i,j) in self._kneighb_pairs(l_unsat, k_unsat, kneighb)) )
                    vv_list = sorted( set(edge_list),  key = self._vv_dist_key)
                    
                    
                else: # k==l:
                    
                    Jkl = jdd[k][k] = jdd[k][k]/2   ## double counting in  nkk for case k=l
    
                    k_unsat = [v for v in h_degree_nodelist[k] if node_residual[v]>0 ]
                                        
                    #### kneighb pairs only
                    edge_list = set( self._kneighb_pairs(k_unsat, k_unsat, kneighb+1) )                    
                    vv_list = sorted( ( (i,j) for (i,j) in edge_list if ((j,i) not in edge_list or i>j) ), key=self._vv_dist_key )
                       
                                    
    
                    
                for (v,w) in vv_list:                                    
                    if (node_residual[v]>0) and (node_residual[w]>0) and not Gout.has_edge(v,w):                                                                                                            
                        if (Jkl<=0) :
                            break                            
                        
                        Gout.add_edge(v,w)
                        
                        node_residual[v] -= 1
                        node_residual[w] -= 1                      
                        Jkl -= 1      
                        
                        
    
                                                                                                                                                                                                                              
                if k>l:
                    jdd[k][l] = jdd[l][k] = Jkl
                else: # k==l
                    jdd[k][l] = jdd[l][k] = Jkl *2
                                                                        
            if p_idx % dot_freq == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
                            
            
        print "\n[1st] kneighb residual stubs: %d/%d" % ( sum( node_residual.values()),  doubledict_sum(jdd) ) 
        
        
    def _secondpass(self, Gout, jdd, h_degree_nodelist,  node_residual, node_degree):

        kl_pairs = [ (i,j) for i in jdd for j in jdd[i] if jdd[i][j]>0 ]
        random.shuffle(kl_pairs)
        
        dot_freq = int( math.ceil(.01*len(kl_pairs)) )
        for p_idx, (k,l) in enumerate(kl_pairs):
            if jdd[k][l]>0 and k>=l:
                if k!=l:
                    
                    Jkl = jdd[k][l]
    
                    
                    ####### throw edges that are possible between unsaturated nodes
                    ##########                
                    
                    k_unsat = [v for v in h_degree_nodelist[k] if node_residual[v]>0 ]
                    l_unsat = [w for w in h_degree_nodelist[l] if node_residual[w]>0 ]                                                                                    
                    
                    ####vv_unconnected = it.product(k_unsat,l_unsat)
                    
                    sz = 3000
                    vv_unconnected = it.product(random.sample(k_unsat, min(sz, len(k_unsat) ) ), \
                                                random.sample(l_unsat, min(sz, len(l_unsat) ) )    )            
                    for (v,w) in vv_unconnected:
                        if node_residual[v]>0 and node_residual[w]>0 and not Gout.has_edge(v,w):
                            if (Jkl==0) :
                                break
                            
                            Gout.add_edge(v,w)  
                            node_residual[v] -= 1
                            node_residual[w] -= 1                      
                            Jkl -= 1               
    
                    
                    ####### trick 1
                    ##########        
                    if Jkl > 0:
                        k_unsat_Lash = Lash([v for v in h_degree_nodelist[k] if node_residual[v]>0 ])
                        l_unsat_Lash = Lash([w for w in h_degree_nodelist[l] if node_residual[w]>0 ])                    
                        
                        all_unsat     =   [v for v in k_unsat_Lash.get_all() for i in range(node_residual[v]) ]
                        all_unsat.extend( [w for w in l_unsat_Lash.get_all() for i in range(node_residual[w]) ] )
    
                        
                        k_all = set(h_degree_nodelist[k])
                        l_all = set(h_degree_nodelist[l])
                        
                        random.shuffle( all_unsat )
                        for va in all_unsat:   
                            if node_residual[va]:  ## va is unsaturated
                                
                                if    node_degree[va] == k and l_unsat_Lash.size(): 
                                    a = k ; a_unsat = k_unsat_Lash
                                    b = l ; b_unsat = l_unsat_Lash ; b_all = l_all
                                elif node_degree[va] == l and k_unsat_Lash.size(): 
                                    a = l ; a_unsat = l_unsat_Lash
                                    b = k ; b_unsat = k_unsat_Lash ; b_all = k_all
                                else :
                                    continue
                                                                
                                vb = b_unsat.get_random()
                                
                                vbprime_list = b_all - set(Gout[va])  ## all nodes of degree "b" not connected to "va"
                                
                                for vbprime in vbprime_list:
                                    
                                    swap_candidates = list( set(Gout[vbprime]) - set(Gout.neighbors(vb)+[vb]) ) ## all neighbors of "vbprime" not connected to "vb"
                                    if len(swap_candidates):
                                        chosen_node = random.choice(swap_candidates)  
                                        
                                        Gout.add_edge(va,vbprime)   # +1 for va
                                        Gout.remove_edge(vbprime, chosen_node)
                                        Gout.add_edge(vb, chosen_node)  # +1 for vb
    
                                        ### net zero for 'chosen_node', net zero for 'vbprime'
                                        
                                        #print "add ", va, "-", vbprime
                                        #print "rem ", vbprime, "-", chosen_node
                                        #print "add ", vb, "-", chosen_node                               
                                        
                                        node_residual[va] -= 1 
                                        node_residual[vb] -= 1
                                        Jkl -= 1             
    
                                        
                                        if not node_residual[va]:
                                            a_unsat.delete(va)
                                        if not node_residual[vb]:
                                            b_unsat.delete(vb)
                                                               
                                        break 
                                    
                            if Jkl==0:
                                break
                    
                    
                    jdd[k][l] = jdd[l][k] = Jkl
                    
                    ####### trick 3
                    ##########     
                    if jdd[k][l] > 0: #
    
                        k_sat   = [v for v in h_degree_nodelist[k] if node_residual[v]==0 ]
                        l_sat   = [w for w in h_degree_nodelist[l] if node_residual[w]==0  ] 
                                        
                        ###vv_unconnected = it.product(k_sat,l_sat)
                        
                        sz = 3000
                        vv_unconnected = it.product(random.sample(k_sat, min(sz, len(k_sat) ) ), \
                                                    random.sample(l_sat, min(sz, len(l_sat) ) )    )            
    
                    
                        
                        for (v,w) in vv_unconnected:
                            if  not Gout.has_edge(v,w) and Gout.degree(v)>0 and Gout.degree(w)>0:
                                
                                v_prime = random.choice( list(Gout[v]) )
                                w_prime = random.choice( list(Gout[w]) )
                                
                                Gout.add_edge(v,w)
    
                                Gout.remove_edge(v,v_prime)
                                Gout.remove_edge(w,w_prime)
                                
                                        
                                node_residual[v_prime] += 1
                                node_residual[w_prime] += 1                      
                                 
                                jdd[k][node_degree[v_prime]] += 1
                                if k!=node_degree[v_prime]:
                                    jdd[node_degree[v_prime]][k] += 1
                                
                                jdd[l][node_degree[w_prime]] += 1
                                if k!=node_degree[w_prime]:
                                    jdd[node_degree[w_prime]][l] += 1
                                
                                jdd[k][l] -= 1              
                                jdd[l][k] -= 1
                                
                                if (jdd[k][l]==0) :
                                    break                                
                    
                                
                elif k==l:
    
                    Jkl = jdd[k][l]
                    
                    
                    ####### throw edges that are possible between unsaturated nodes
                    ##########                
                    
                    k_unsat = [v for v in h_degree_nodelist[k] if node_residual[v]>0 ]
    
                    ###vv_unconnected = it.product(k_unsat,k_unsat)
                    
                    sz = 3000
                    vv_unconnected = it.product(random.sample(k_unsat, min(sz, len(k_unsat) ) ), \
                                                random.sample(k_unsat, min(sz, len(k_unsat) ) )    )            
                    
                    
                    
                    for (v,w) in vv_unconnected:
                        if node_residual[v]>0 and node_residual[w]>0 and not Gout.has_edge(v,w) and v!=w:                        
                            if (Jkl==0) :
                                break
                            
                            Gout.add_edge(v,w)
                            
                            
                            node_residual[v] -= 1
                            node_residual[w] -= 1                      
                            Jkl -= 1               
                                
                            
                    
                    ####### trick 1
                    ##########        
                            
                    if Jkl > 0:
                        k_unsat_Lash = Lash( [v for v in h_degree_nodelist[k] if node_residual[v]>0 ]   )
                        
                        all_unsat     =   [v for v in k_unsat_Lash.get_all() for i in range(node_residual[v]) ]
    
                        k_all = set(h_degree_nodelist[k])
                                                
                        random.shuffle( all_unsat )
                        for va in all_unsat:   
                            if node_residual[va]:  ## va is unsaturated
                                
                                if node_residual[va]==1:
                                    k_unsat_Lash.delete(va)
                                    
                                if not k_unsat_Lash.size():
                                    continue
                                                                                            
                                vb = k_unsat_Lash.get_random()
                                
                                vbprime_list = k_all - set(Gout.neighbors(va) + [va])  ## all nodes of degree "k" not connected to "va"
                                
                                for vbprime in vbprime_list:
                                    
                                    swap_candidates = list( set(Gout[vbprime]) - set(Gout.neighbors(vb) + [vb] ) )  ## all neighbors of "vbprime" not connected to "vb"
                                    if len(swap_candidates):
                                        chosen_node = random.choice(swap_candidates)  
                                        
                                        Gout.add_edge(va,vbprime)   # +1 for va
                                        Gout.remove_edge(vbprime,chosen_node)
                                        Gout.add_edge(vb,chosen_node)  # +1 for vb
    
                                        ### net zero for 'chosen_node', net zero for 'vbprime'
                                        
                                        #print "add ", va, "-", vbprime
                                        #print "rem ", vbprime, "-", chosen_node
                                        #print "add ", vb, "-", chosen_node                               
                                        
                                        node_residual[va] -= 1 
                                        node_residual[vb] -= 1
                                        Jkl -= 1             
                                        
                                        #print Jkl
                                        
                                        if not node_residual[va]:
                                            k_unsat_Lash.delete(va)
                                        if not node_residual[vb]:
                                            k_unsat_Lash.delete(vb)
                                                               
                                        break 
                                    
                            if Jkl==0:
                                break
    
                    jdd[k][l] =  Jkl
    
                    ####### trick 3
                    ##########     
                    if jdd[k][l] > 0: #
    
                        k_sat   = [v for v in h_degree_nodelist[k] if node_residual[v]==0 ]                                    
                            
                        ###vv_unconnected = it.product(k_sat,k_sat)
                        sz = 3000
                        vv_unconnected = it.product(random.sample(k_sat, min(sz, len(k_sat) ) ), \
                                                    random.sample(k_sat, min(sz, len(k_sat) ) )    )            
                         
                        for (v,w) in vv_unconnected:
                            if  not Gout.has_edge(v,w) and Gout.degree(v)>0 and Gout.degree(w)>0 and v!=w:
                                
                                v_prime = random.choice( list(Gout[v]) )
                                w_prime = random.choice( list(Gout[w]) )
                                
                                Gout.add_edge(v,w)
    
                                Gout.remove_edge(v,v_prime)
                                Gout.remove_edge(w,w_prime)
                                
                                        
                                node_residual[v_prime] += 1
                                node_residual[w_prime] += 1                      
                                 
                                jdd[k][node_degree[v_prime]] += 1
                                if k!=node_degree[v_prime]:
                                    jdd[node_degree[v_prime]][k] += 1
                                                            
                                jdd[k][node_degree[w_prime]] += 1 
                                if k!=node_degree[w_prime]:
                                    jdd[node_degree[w_prime]][k] += 1
                                
                                jdd[k][k] -= 1              
                                
                                if (jdd[k][k]==0) :
                                    break
                                
                            
            if p_idx % dot_freq == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
                    
        
        print "\n2nd pass residual stubs: %d/%d" % ( sum( node_residual.values()),  doubledict_sum(jdd) ) 
        
    def construct_triangles_2K(self):
        '''
        2K construction algorithm described in below paper:
        M. Gjoka, M. Kurant, A. Markopoulou
        2.5K-Graphs: from Sampling to Generation,
        under submission
        '''

        if not self.JDD_target:
            print "JDD not defined"
            return
            
        
        
        print "\n2k triangles construction started"
        Gout = nx.Graph()                        
    
        ##########  Initialize data structures
        ################################################################
        
        jdd = copy.deepcopy(self.JDD_target)    
         
        dd = degree_dist(jdd)  
        nNodes = sum(dd.values())
    

        nodeid = 0
        h_degree_nodelist = {}
        for degree,numNodes in dd.iteritems():
            h_degree_nodelist[degree] = range(nodeid, nodeid+int(numNodes))
            nodeid += int(numNodes)
        
        node_residual = {}
        node_degree = {}
        self.node_coord = {}
        
        for degree,nodelist in h_degree_nodelist.iteritems():
            for v in nodelist:
                node_residual[v] = degree
                node_degree[v] = degree
                self.node_coord[v] = random.random()
                if v not in Gout:
                    Gout.add_node(v)
            h_degree_nodelist[degree].sort( key=self.node_coord.__getitem__)
       
        ##########  Construct graph
        ################################################################
        kneighb = 3
        batch_size = 5*10**5
        npasses = 0
        

        ##### All possible node pairs
        ##### O(n^2), slow for big graphs
        #self.firstpass_allpair(Gout, kneighb, jdd , h_degree_nodelist, node_residual, node_degree)        

        
        ##### Smart all node pairs
        ##### approximate implementation of "all possible node pairs" in O(kneigb*n)
        self._firstpass_allpair_smart(Gout, batch_size, jdd , h_degree_nodelist, node_residual, node_degree)        
        self._firstpass_kneighb(Gout, kneighb, jdd , h_degree_nodelist, node_residual)    
        
        
        for i in jdd:
            for j in jdd[i]:
                if (i==j):
                    jdd[i][j] /= 2    
        
        while (npasses<20) and doubledict_sum(jdd)>0:
            print "\n-------\n[%d]" % ( npasses )
            self._secondpass(Gout, jdd, h_degree_nodelist,  node_residual, node_degree)
            npasses += 1
        
        if Gout.number_of_nodes():
            print "--Edges:%d  Nodes:%d  Avg:%.2f" % (Gout.number_of_edges(), Gout.number_of_nodes(), \
                                                      2.*Gout.number_of_edges()/Gout.number_of_nodes()  ) 
        
        self.Gdk = Gout
    
    def construct_simple_2K(self):
        '''
        2K construction algorithm described in below paper:
        I. Stanton and A. Pinar, 
        "Constructing and sampling graphs with prescribed joint degree distribution using Markov Chains ", 
        submitted to ACM Journal of Experimental Algorithmics 
        '''
        
        if not self.JDD_target:
            print "JDD not defined"
            return
            
        
        print "\n2k simple construction started"
        Gout = nx.Graph()
        
        ##########  Initialize data structures
        ################################################################        
        
        jdd = copy.deepcopy(self.JDD_target)    
        
        ## Get number of nodes of each degree  
        dd = degree_dist(jdd) 
        nNodes = sum(dd.values())
    
        
        nodeid = 0
        h_degree_nodelist = {}
        for degree,numNodes in dd.iteritems():
            h_degree_nodelist[degree] = range(nodeid, nodeid+int(numNodes))
            nodeid += int(numNodes)

    
        node_residual = {}
        for degree,nodelist in h_degree_nodelist.iteritems():
            for v in nodelist:
                node_residual[v] = degree
        
        ##########  Construct graph
        ################################################################ 
        
        kl_pairs = [ (i,j) for i in jdd for j in jdd[i] ]
        random.shuffle(kl_pairs)
                    
        dot_freq = int( math.ceil(.01*len(kl_pairs)) )
        for p_idx, (k,l) in enumerate(kl_pairs):
            if k>=l:
                if (k!=l) and (jdd[k][l]>0):
                    a = int(jdd[k][l] % dd[k])
                    b = int(jdd[k][l] % dd[l])
                    
                    aseq = [int(jdd[k][l]/dd[k] + 1)]*a 
                    if int(jdd[k][l]/dd[k])>=1:
                        aseq = aseq + [int(jdd[k][l]/dd[k])]*int(dd[k]-a)
                                        
                    bseq = [int(jdd[k][l]/dd[l] + 1)]*b 
                    if int(jdd[k][l]/dd[l])>=1:
                        bseq = bseq + [int(jdd[k][l]/dd[l])]*int(dd[l]-b)
                        
                    
                    bp_sg = nx.bipartite_havel_hakimi_graph(aseq,bseq,create_using=nx.Graph())
                    
                    a_map = dict( (v,h_degree_nodelist[k][idx]) for idx, (v,attr) in enumerate(bp_sg.node.items()) \
                                  if (attr['bipartite']==0) and bp_sg.degree(v)>0 )
                    
                    b_map = dict( (v,h_degree_nodelist[l][idx-len(aseq)]) for idx, (v,attr) in enumerate(bp_sg.node.items()) \
                                  if (attr['bipartite']==1) and bp_sg.degree(v)>0 )
                                                        
    
                    for (v,w) in bp_sg.edges_iter():
                        if not (Gout.has_edge( a_map[v], b_map[w] ) ):
                            Gout.add_edge( a_map[v], b_map[w] )
                            node_residual[ a_map[v] ]  -= 1
                            node_residual[ b_map[w] ]  -= 1
                                            
                    jdd[k][l] = jdd[l][k] = 0
                    
                    h_degree_nodelist[k].sort(key=node_residual.get, reverse=1)
                    h_degree_nodelist[l].sort(key=node_residual.get, reverse=1)
                    
                elif (k==l) and (jdd[k][l]>0):
                    c = (int(jdd[k][l] % dd[k] ) )
                    
                    cseq = [int(jdd[k][l]/dd[k] + 1)]*c 
                    if int(jdd[k][l]/dd[k])>=1:
                        cseq = cseq + [int(jdd[k][l]/dd[k])]*int(dd[k]-c)
    
                    sg = nx.havel_hakimi_graph(cseq)
                    
                    c_map = dict( (v,h_degree_nodelist[k][idx]) for idx, v in enumerate(sg) \
                                  if sg.degree(v)>0 )
                    
                    for (v,w) in sg.edges_iter():
                        if not (Gout.has_edge( c_map[v], c_map[w] ) ):
                            Gout.add_edge( c_map[v], c_map[w])
                            node_residual[ c_map[v] ]  -= 1
                            node_residual[ c_map[w] ]  -= 1
                        
                    jdd[k][l] = 0
                                    
                    h_degree_nodelist[k].sort(key=node_residual.get, reverse=1)
    
            if p_idx % dot_freq == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
                    
        
            
        print "\n--Residual stubs: %d/%d" % ( sum( node_residual.values()),  doubledict_sum(jdd) ) 
        print "--Edges:%d  Nodes:%d  Avg:%.2f" % (Gout.number_of_edges(), Gout.number_of_nodes(), \
                                                2.*Gout.number_of_edges()/Gout.number_of_nodes()  ) 
        
        self.Gdk = Gout

        
        

        
    def construct_dkseries_2K(self):
        '''
        2K construction algorithm described in below paper:
        P. Mahadevan, D. Krioukov, K. Fall, and A. Vahdat,
        "Systematic Topology Analysis and Generation Using Degree Correlations",
        SIGCOMM '06        
        '''
        niter = 10

        if not self.JDD_target:
            print "JDD not defined"
            return
            
        
        print "2k dkseries construction started"
        Gout = nx.Graph()
        
        ##########  Initialize data structures
        ################################################################        
        
        nkk = copy.deepcopy(self.JDD_target)    
        
        
        stubs = []    
    
        ## Get number of nodes of each degree  (nk)
        nk = degree_dist(nkk)
        nNodes = sum(nk.values())
    

        nodeid = 0
        h_degree_nodelist = {}
        for degree,numNodes in nk.iteritems():
            h_degree_nodelist[degree] = range(nodeid, nodeid+int(numNodes))
            nodeid += int(numNodes)
    
    
        stubid = 0
    
        degree_stub = {}
        for degree,numNodes in nk.iteritems():
            for n in range( int(numNodes)): ## for every node of this degree
                nodeid = h_degree_nodelist[degree].pop(0)
                for d in range(degree): ## create as many stubs as the degree values
                    s = Stub(degree=degree,nodeid=nodeid)
                    stubs.append(s)
    
                    degree_stub.setdefault(degree,[]).append(stubid)
                    stubid += 1
    
            if numNodes>0:
                random.shuffle( degree_stub[degree] )
    
        print "%d stubs to add" % len(stubs)
        available = [1]*len(stubs)
        
        unconnected_stubs_prev = -1
        j = 0
        while (j<niter) and unconnected_stubs_prev != sum(available):
            randomstubs =   [i for i in range( len(stubs) ) if available[i] ]
            random.shuffle( randomstubs )
    
            unconnected_stubs_prev = sum(available)
            self._dkgen2k_connectstubs(stubs, randomstubs, degree_stub, available, nkk, Gout)            
            print "\nResidual stubs: %d\n" % sum(available), 
    
        
    
        unusedStubs = [stubs[i] for i in range( len(stubs) ) if available[i] ]
          
        
        print "--Edges:%d  Nodes:%d  Avg:%.2f" % (Gout.number_of_edges(), Gout.number_of_nodes(), \
                                            2.*Gout.number_of_edges()/Gout.number_of_nodes()  ) 

        self.Gdk = Gout
        

    def _compute_kchange(self, G, G_degree, edges_change, shared_vv):
        ### optimize
        k_change = {}
    
        hpastNeighb = defaultdict(set)
        for v,w,change in edges_change:
            if change<0: # edge will be removed
                hpastNeighb[v].add(w)
                hpastNeighb[w].add(v)            
        
        for v,w,change in edges_change:
            if change>0: # add edge
                sharedpartners = set(G[v]) & set(G[w]) - hpastNeighb[v] - hpastNeighb[w]
            else: # remove edge
                sharedpartners = shared_vv.get(v,{}).get(w,set()) - hpastNeighb[v] - hpastNeighb[w]
            
            kv = G_degree[v]
            kw = G_degree[w]
    
            k_change[kv] = k_change.get(kv,0) + 2*len(sharedpartners)*change
            k_change[kw] = k_change.get(kw,0) + 2*len(sharedpartners)*change
    
            degree_count = countOccur( (G_degree[i] for i in sharedpartners ) )
            for ki,c in degree_count.items():
                k_change[ki] = k_change.get(ki,0) + 2*change*c
    
    
        return k_change
    
    
    def _compute_vvchange(self, G, edges_change, shared_vv):
        vv_change = {}
    
        hpastNeighb = defaultdict(set)
        for v,w,change in edges_change:
            if change<0: # edge will be removed
                hpastNeighb.setdefault(v,set()).add(w)
                hpastNeighb.setdefault(w,set()).add(v)
    
        for v,w,change in edges_change:
            if change>0: # add edge
                vv_change[v,w,change] = set(G[v]) & set(G[w]) - hpastNeighb[v] - hpastNeighb[w]
            else: # remove edge
                vv_change[v,w,change] = shared_vv.get(v,{}).get(w,set()) - hpastNeighb[v] - hpastNeighb[w]
                            
            vv_change[w,v,change] = vv_change[v,w,change]
                    
            
            for node in vv_change[v,w,change]:
                vv_change[node,v,change] = set([w])
                vv_change[node,w,change] = set([v])
                
                vv_change[v,node,change] = vv_change[node,v,change]
                vv_change[w,node,change] = vv_change[node,w,change]
    
    
        return vv_change
    
        
    def _changescore_CCk(self, k_change, diff_ktri, normFunc=defaultdict(lambda:1)):
        
        score_old = 0
        score_new = 0
        
        for kv in k_change:
            val = diff_ktri[kv]/normFunc[kv]     
            score_old += val**2                
            
            val += float(k_change[kv])/normFunc[kv]
            score_new += val**2            
            
        return score_new, score_old
    

    def _make_changes(self, G, G_degree, edges_change, edges_deg):
        ### it is assumed that appropriate edge additions and deletions
        ### are scheduled together such that nkk is preserved
    
    
        ## add/remove edges from G and edges_deg
        for v,w,change in edges_change:
            if change>0 :  # add edge
                G.add_edge(v,w)
    
                edges_deg[ G_degree[v] ].add( (v,w))
                edges_deg[ G_degree[w] ].add( (w,v))            
            else: # remove edge
                G.remove_edge(v,w)
    
                edges_deg[ G_degree[v] ].delete( (v,w))
                edges_deg[ G_degree[w] ].delete( (w,v))            
    

                
                
    def _update_ktri(self, k_change, k_tri):
        for (kv,change) in k_change.items():
            k_tri[kv] = k_tri.get(kv,0) + change

    def _update_vvshared(self, vv_change, shared_vv):
        ## shp -> set of sharedpartners for edge "v-w"
        for (v,w,change),shp in vv_change.items():
            if change>0:
                shared_vv.setdefault(v,{}).setdefault(w, set()).update( shp )
            else:
                shared_vv.setdefault(v,{}).setdefault(w, set()).difference_update( shp )
            
            
            
        
    def mcmc_improved_2_5_K(self, nmae_threshold=0.03, is_init=False):  
        ### resumable  function i.e. inside ipython
        
        def norm_nshared((v,w)):
            return len( self.vv_shared.get(v,{}).get(w,set()) ) / (1.* min(G_degree[v],G_degree[w]) )            
        
        print "Improved double edge swaps"
        
        ##############################
        ####Initialize data structures
        ##############################
        
        if (not self.G) or (is_init): 
            print "Graph copy under way"
            self.G = nx.Graph(self.Gdk)      
            
            print "Data structure computation under way"                        
            self.vv_shared, self.k_tri,  self.k_edges, self.nk = self.graph_properties(self.G)    
            
            self.step_Count, self.swap_Count, self.swap_Count_Prev = 0, 0, -1        
    
            
            
        CCK_target = ktri_to_CCk(self.K_TRI_target, self.nk)                        
    
        nMaxSteps, attempts, nkeep = 10**9, 30, 10        
        
        start_time = curr_time = time.time()
        print "\n[%s]" % (time.ctime())
    

        #####################################################################
        ##### MCMC 
        #####################################################################
    
        norm_cc =  dict(  [ (k, self.nk[k]*k*(k-1)) if k>1 else (1,1) for k in self.nk] )    
        diff_ktri = compute_diff_k( self.k_tri, self.K_TRI_target)
        klist = sorted(i for i in diff_ktri if i>1)        
        G_degree = self.G.degree() 
    
        while self.step_Count<nMaxSteps:
                        
            cdf = nx.utils.cumulative_distribution( [abs(diff_ktri[k])/norm_cc[k]  for k in klist ] )
            for kidx in nx.utils.discrete_sequence(5,cdistribution=cdf):
                self.step_Count += 1
                
                k1 = klist[kidx]                
                
                ## pick at least one edge that increases/decreases shared partners of given degree    
        
                if diff_ktri[k1]<0: #below target
                    ## pick one edge with LEAST shared partners                
                    candidates_edge12 = heapq.nsmallest(nkeep, self.k_edges[k1].get_randomlist(attempts), key=norm_nshared )  
                else:  #above target
                    candidates_edge12 = random.sample(self.k_edges[k1].get_randomlist(attempts), nkeep)
                                    
                v1, v2 = random.choice(candidates_edge12)            
        
                k2 = G_degree[v2]                
        
                candidates_edge34 =  [ (v3,v4) for (v3,v4) in self.k_edges[k2].get_randomlist(attempts) \
                                       if (v3!=v2) and (v1!=v3) and (v2!=v4) \
                                       and (not self.G.has_edge(v1,v3)) and (not self.G.has_edge(v2,v4)) ]
        
                
                if diff_ktri[k1]<0: #below target
                    candidates_edge34 = heapq.nsmallest(nkeep, candidates_edge34, key=norm_nshared )                                                        
                else: #above target
                    candidates_edge34 = random.sample(candidates_edge34, min(nkeep,len(candidates_edge34)))
                      
        
                cases = it.product( [(v1,v2)], candidates_edge34)
        
                for ((v1,v2),(v3,v4)) in cases:
        
                    ######## estimate shared partners
                    ####
                    ## 
                    edges_change = [(v1,v2,-1),(v3,v4,-1),(v1,v3,1),(v2,v4,1)]
                    k_change = self._compute_kchange(self.G, G_degree, edges_change, self.vv_shared)
                    
                    distance_new, distance_old = self._changescore_CCk(k_change, diff_ktri, normFunc=norm_cc)
                    
        
                    if distance_new < distance_old: ## successful double edge swap
                        self.swap_Count +=1
                        self._make_changes(self.G, G_degree, edges_change, self.k_edges)
                        self._update_ktri(k_change, self.k_tri)
                        self._update_ktri(k_change, diff_ktri)
                        
                        vv_change = self._compute_vvchange(self.G, edges_change, self.vv_shared)      
                        self._update_vvshared(vv_change, self.vv_shared)
                        
                        break
        
        
                #######################
                ### Progress update messages
                #######################        
                if self.step_Count % self.update_Freq == 0:
                            
                    duration = time.time()-curr_time
                    avgCC = clustcoeff(self.k_tri, self.nk)
                    nmae = singledict_nmae( CCK_target, ktri_to_CCk(self.k_tri, self.nk) )
                    
                    print "[%.1f][%.2f] #Swaps:%d, C_avg:%.4f, NMAE:%.4f, #Triangles:%d" % \
                          (duration, (self.swap_Count-self.swap_Count_Prev)/duration, \
                           self.swap_Count, avgCC, nmae, sum(self.k_tri.values()))
        
                    self.swap_Count_Prev, curr_time = self.swap_Count, time.time()                    
                        
                    if nmae<nmae_threshold:
                        self.step_Count = nMaxSteps
                    
                    
        print "Total time:%d" % (time.time() - start_time)
        
    
        
    def mcmc_random_2_5_K(self, nmae_threshold=0.03, init=False):
        ### resumable  function i.e. inside ipython
        
        print "Random double edge swaps"
        
        ##############################
        ####Initialize data structures
        ##############################
        
        if (not self.G) or (is_init): 
            print "Graph copy under way"
            self.G = nx.Graph(self.Gdk)      
            
            print "Data structure computation under way"                        
            self.vv_shared, self.k_tri,  self.k_edges, self.nk = self.graph_properties(self.G)    
            
            self.step_Count, self.swap_Count, self.swap_Count_Prev = 0, 0, -1        
    
            
            
        CCK_target = ktri_to_CCk(self.K_TRI_target, self.nk)                        
    
        nMaxSteps, attempts, nkeep = 10**9, 30, 10        
        
        start_time = curr_time = time.time()
        print "\n[%s]" % (time.ctime())
        
                           
        #####################################################################
        ##### MCMC 
        #####################################################################
    
        norm_cc =  dict(  [ (k, self.nk[k]*k*(k-1)) if k>1 else (1,1) for k in self.nk] )    
        diff_ktri = compute_diff_k( self.k_tri, self.K_TRI_target)
        klist = sorted(i for i in diff_ktri if i>1)        
        G_degree = self.G.degree() 

        while self.step_Count<nMaxSteps:
                            
            cdf = nx.utils.cumulative_distribution( [abs(diff_ktri[k])/norm_cc[k]  for k in klist ] )
            for kidx in nx.utils.discrete_sequence(5,cdistribution=cdf):
                self.step_Count += 1
                
                k1 = klist[kidx]                                           
    
                #### pick random edge that has a node with degree k1 
                
                v1, v2 = random.choice( self.k_edges[k1].get_randomlist(attempts) )
        
                k2 = G_degree[v2]
        
                candidates_edge34 =  [ (v3,v4) for (v3,v4) in self.k_edges[k2].get_randomlist(attempts) \
                                       if (v3!=v2) and (v1!=v3) and (v2!=v4) \
                                       and (not self.G.has_edge(v1,v3)) and (not self.G.has_edge(v2,v4)) ]
             
                      
                cases = it.product( [(v1,v2)], candidates_edge34[:nkeep])
        
                           
                for ((v1,v2),(v3,v4)) in cases:
                                                
                    ######## estimate shared partners
                    ####
                    edges_change = [(v1,v2,-1),(v3,v4,-1),(v1,v3,1),(v2,v4,1)]
                    k_change = self._compute_kchange(self.G, G_degree, edges_change, self.vv_shared)
                    
                    distance_new, distance_old = self._changescore_CCk(k_change, diff_ktri, normFunc=norm_cc)
                    
        
                    if distance_new < distance_old: ## successful double edge swap
                        
                        self.swap_Count +=1
                        self._make_changes(self.G, G_degree, edges_change, self.k_edges)
                        self._update_ktri(k_change, self.k_tri)
                        self._update_ktri(k_change, diff_ktri)
                        
                        vv_change = self._compute_vvchange(self.G, edges_change, self.vv_shared)      
                        self._update_vvshared(vv_change, self.vv_shared)
                        
                        break
        
                    
        
                    
                #######################
                ### Progress update messages
                #######################        
                if self.step_Count % self.update_Freq == 0:
                            
                    duration = time.time()-curr_time
                    avgCC = clustcoeff(self.k_tri, self.nk)
                    nmae = singledict_nmae( CCK_target, ktri_to_CCk(self.k_tri, self.nk) )
                    
                    print "[%.1f][%.2f] #Swaps:%d, C_avg:%.4f, NMAE:%.4f, #Triangles:%d" % \
                          (duration, (self.swap_Count-self.swap_Count_Prev)/duration, \
                           self.swap_Count, avgCC, nmae, sum(self.k_tri.values()))
        
                    self.swap_Count_Prev, curr_time = self.swap_Count, time.time()                    
                        
                    if nmae<nmae_threshold:
                        self.step_Count = nMaxSteps
                        
                        
        print "Total time:%d" % (time.time() - start_time)
    
    def save_graphs(self,prefix_name):
        if self.Gdk:
            fname = '%s_2K.edges.gz' % prefix_name
            print "Saving 2K at \"%s\"" % fname
            nx.write_edgelist(self.Gdk, fname)
            
        if self.G:
            fname = '%s_2.5K.edges.gz' % prefix_name
            print "Saving 2.5K at \"%s\"" % fname
            nx.write_edgelist(self.G, fname)            
                    
def test_generation():
    mygen = Generation()


    #### Estimate 
    fname = "Caltech36.mat"    
    myest = Estimation()    
    myest.load_graph(fname)
    
    print "Calculating JDD, CCK"
    myest.calcfull_JDD()    
    myest.calcfull_CCK()
    
    #### set target JDD, KTRI
    mygen.set_JDD( myest.get_JDD('full') )
    mygen.set_KTRI( myest.get_KTRI('full') )
    
    ####### pick 2K construction algorithm
    mygen.construct_triangles_2K()
    #mygen.construct_simple_2K()    
    #mygen.construct_dkseries_2K()
    
    ###### pick 2.5K MCMC targetting algorithm
    mygen.mcmc_improved_2_5_K()
    #mygen.mcmc_random_2_5_K()
    
    return mygen
    
if __name__ == "__main__":
        
    
    print "Testing Generation module"
    mygen = test_generation()
     