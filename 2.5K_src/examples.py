'''
Examples of usage for classes Estimation and Generation.
'''

from Estimation import Estimation
from Generation import Generation

__author__ = """Minas Gjoka"""

if __name__ == "__main__":
    
    '''
    You can find below topologies at
    http://www.minasgjoka.com/2.5K
    '''
    fname = "UCSD34.mat"
    #fname = "Harvard1.mat"
    #fname = "Facebook-New-Orleans.edges.gz"
    #fname = 'soc-Epinions1.edges.gz'
    #fname = "email-Enron.edges.gz"
    #fname = "as-caida20071105.edges.gz"    
    
    run_case = 1
    
    ###### Full graph - 2K with triangles + Improved MCMC 
    if run_case == 1:
        myest = Estimation()
        myest.load_graph(fname)
        myest.calcfull_CCK()
        myest.calcfull_JDD()
        
        mygen = Generation()
        mygen.set_JDD( myest.get_JDD('full') )
        mygen.set_KTRI( myest.get_KTRI('full') ) 
        
        mygen.construct_triangles_2K()
        mygen.mcmc_improved_2_5_K(nmae_threshold=0.03)
        mygen.save_graphs('%s_2KT+ImpMCMC_Full' % fname)
    #######################################################
    ###### Full graph - 2K simple +  MCMC  
    elif run_case == 2:
        myest = Estimation()
        myest.load_graph(fname)
        myest.calcfull_CCK()
        myest.calcfull_JDD()
        
        mygen = Generation()
        mygen.set_JDD( myest.get_JDD('full') )
        mygen.set_KTRI( myest.get_KTRI('full') ) 
        
        mygen.construct_simple_2K()
        mygen.mcmc_random_2_5_K(nmae_threshold=0.03)
        mygen.save_graphs('%s_2Ksimple+MCMC_Full' % fname)
    #######################################################
    ###### 30% sample - 2K with triangles + Improved MCMC 
    elif run_case == 3:
        p_sample = 0.3
        myest = Estimation()
        myest.load_graph(fname)
        myest.sample('rw', p_sample)
        myest.estimate_JDD()
        myest.estimate_CCK()
        
        mygen = Generation()
        mygen.set_JDD( myest.get_JDD('realizable') )
        mygen.set_KTRI( myest.get_KTRI('estimate') ) 
        
        mygen.construct_triangles_2K()
        mygen.mcmc_improved_2_5_K(nmae_threshold=0.05)
        mygen.save_graphs('%s_2KT+ImpMCMC_%.2fsample' % (fname, p_sample))
    #######################################################
    ###### 30% sample - 2K simple +  MCMC  
    elif run_case == 4:
        p_sample = 0.3
        myest = Estimation()
        myest.load_graph(fname)
        myest.sample('rw', p_sample)
        myest.estimate_JDD()
        myest.estimate_CCK()
        
        mygen = Generation()
        mygen.set_JDD( myest.get_JDD('realizable') )
        mygen.set_KTRI( myest.get_KTRI('estimate') ) 
        
        mygen.construct_simple_2K()
        mygen.mcmc_random_2_5_K(nmae_threshold=0.05)
        mygen.save_graphs('%s_2Ksimple+MCMC_%.2fsample' % (fname, p_sample))        
        