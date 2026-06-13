import numpy as np
from scipy.optimize import toms748
from tqdm import tqdm
import time
from dataclasses import dataclass

@dataclass
class Samples:
    u: np.ndarray
    theta: np.ndarray
    r_batch: np.ndarray
    sample_log_density: np.ndarray
    
    def extend(self, other):
        self.u = np.concatenate([self.u, other.u], axis=0)
        self.theta = np.concatenate([self.theta, other.theta], axis=0)
        self.r_batch = np.concatenate([self.r_batch, other.r_batch], axis=0)
        self.sample_log_density = np.concatenate([self.sample_log_density, other.sample_log_density], axis=0)
    def filter(self, mask):
        return Samples(
            u=self.u[mask],
            theta=self.theta[mask],
            r_batch=self.r_batch[mask],
            sample_log_density=self.sample_log_density[mask]
        )   


class sampling:
    def sampling_f_r_new(self,density,batch_size=256,alpha=0.1,thresh_acceptance=0.1,angle_importance=np.pi/10,tau=0.01):

        """
        alpha: if we want k samples, we will check k+alpha%k samples. eg: alpha=0.01
        batch_size: no. of samples procced at a time.
        m: no. of directions to open cone for importance sampling.
        
        - if I find the emperical_log_f_max higher than previous one, then we do retrospective prunning. 
        
        - Will do this newer sample search for some number with respect to remaining numbner
        of samples. something like --- 
        (remaining_no_samples)+(remaining_no_samples)*alpha
        
        - Might work, since there is less chance of finding bigger_f_max. if i do then 
        i will only have to see ONLY previous accepted samples (not rejected ones since they are 
        even smaller.)
        
        density: log(f(r)) + (d-1) log(r)
        """
        rng=np.random.default_rng()
        t_main=time.perf_counter()
        def batch_sampling(no_samples, first,theta_sampling=False,importance_orthants=None, importance_directions=None, importance_mass=None,old_log_f_max=None):   
            top_mass_theta,top_orthants,top_masses=[],[],[] #shifted this from outside batch_sampling function to here.
            maximums_log=[]
            running_mean,running_variance,n=0,0,0
            possible_samples=Samples(u=np.array([]),theta=np.empty((0,self.d)),r_batch=np.array([]),sample_log_density=np.array([]))
            for _ in range(np.maximum(round(no_samples/batch_size)+round((no_samples/self.k)*alpha),1)):
                if theta_sampling:
                    # will do rejection sampling on orthants to choose one for each batch.
                    selected=False
                    number_of_orthants=len(importance_orthants)
                    max_prob=importance_mass[0]
                    # print(len(importance_orthants))  
                    while not selected:
                        index=rng.integers(number_of_orthants-1)
                        selected= max_prob*(rng.uniform(0,1))<=importance_mass[index]
                        if selected:
                            theta_batch=self.orthant_theta_generator(importance_orthants[index],batch_size)
                            # print(importance_mass)
                            R_batch=self.R(theta_batch)
                            a_batch,b_batch,log_f_max_batch,total_mass_batch=self.importance_r(density,R_batch,theta_batch)
                            new_orthant_avg_mass=np.average(total_mass_batch)
                            sampled_r_batch=np.random.uniform(a_batch,b_batch)
                            #dividing density with mass of that orthant, aka the bias mitigation step.
                            density_vals=density(sampled_r_batch,theta_batch)-np.log(importance_mass[index])
                            importance_mass[index]=new_orthant_avg_mass
                else:
                    t0=time.perf_counter()
                    theta_batch=self.theta_generation(batch_size)
                    R_batch=self.R(theta_batch)
                    t1=time.perf_counter()
                    print(t1-t0,'theta')
                    theta_dummy=self.theta_generation(1)
            
                    _dummy,_2,_3,_4=self.importance_r(density,self.R(theta_dummy),theta_dummy)
                    t2=time.perf_counter()
                    print(t2-t1,'dummy')
                    a_batch,b_batch,log_f_max_batch,total_mass_batch=self.importance_r(density,R_batch,theta_batch)
                #use this total_mass_batch for the running mean and variance calculation.
                    t3=time.perf_counter()
                    print(t3-t2,'importance_r') 
                    sampled_r_batch=np.random.uniform(a_batch,b_batch)
                    density_vals=density(sampled_r_batch,theta_batch)
                    
                u_batch=np.log(np.random.uniform(0,1,len(theta_batch)))
                batch_samples=Samples(u=u_batch,theta=theta_batch,r_batch=sampled_r_batch,sample_log_density=density_vals)
                possible_samples.extend(batch_samples)
                maximums_log.append(np.max(log_f_max_batch))
                print(np.max(log_f_max_batch))
                print(np.max(maximums_log),'array')
                #for importance in theta
                if first:
                    #update running_mean and variance here.
                    m=len(total_mass_batch)
                    running_mean=(n*running_mean+np.sum(total_mass_batch))/(n+m)
                    n+=m
                    thresh_angle=2*angle_importance
                    top_m_orthants_batch,top_m_theta_batch,top_m_mass_batch=self.away_thetas_batch(theta_batch,total_mass_batch,thresh_angle,tau,batch=True)
                    top_mass_theta.extend(top_m_theta_batch)
                    top_masses.extend(top_m_mass_batch)
                    top_orthants.extend(top_m_orthants_batch)
                    # print(len(top_m_orthants_batch))
                    # print(len(top_orthants))
                    # print(top_m_theta_batch,top_m_mass_batch)
            
            emperical_log_f_max=np.max(np.array(maximums_log))
            if old_log_f_max is not None and emperical_log_f_max <= old_log_f_max:
                print(emperical_log_f_max,'loacal oneee')
                print(np.max(np.array(maximums_log)))
                emperical_log_f_max=old_log_f_max
                
            mask = emperical_log_f_max+possible_samples.u<possible_samples.sample_log_density
            accepted=possible_samples.filter(mask)
            rejected=possible_samples.filter(~mask)
            # print(len(accepted.u),len(rejected.u)) 
             
            # will have to check again for away directions before adding to the global list.
            if first:
                # change tau here. 
                # tau = factor / mean, need to make sure it is fine for both batch wise and global.
                top_m_orthants,top_m_theta, corresponding_masses=self.away_thetas_batch(np.array(top_mass_theta),np.array(top_masses),thresh_angle,tau,global_mean=running_mean,orthants_batch=np.array(top_orthants),batch=False)
                # print(len(top_m_orthants))
                return accepted,rejected,emperical_log_f_max,top_m_orthants,top_m_theta,corresponding_masses
            return accepted,rejected,emperical_log_f_max,importance_orthants,importance_mass


        with tqdm(total=self.k,unit=' accepted samples ') as pbar:
            previous=0

            accepted,rejected,emperical_log_f_max, top_m_orthants,top_m_theta, top_masses=batch_sampling(self.k+round(self.k*alpha),first=True)
            # print(emperical_log_f_max) 
            accepted_count=len(accepted.u)
            pbar.update(accepted_count)
            rejected_count=len(rejected.u)
            theta_sampling=False
            importance_directions=None
            importance_mass=None
            importance_orthants=None
            acceptance_ratio=accepted_count/(accepted_count+rejected_count)
            # we put condtion here for importance theta, if i have acceptance ratio
            # smaller than 'threshold' then will go for importance theta sampling. 
            print(acceptance_ratio,'acceptance_ratio',thresh_acceptance)
            if acceptance_ratio < thresh_acceptance:
                # do importance theta sampling
                theta_sampling=True
                importance_directions=top_m_theta
                importance_mass=top_masses
                importance_orthants=top_m_orthants
                print('switching to importance theta.')
            while (self.k-accepted_count)>0:
                
                remaining=self.k-accepted_count
                new_batch_size=remaining+round(remaining*alpha)
                new_acc,new_reject,new_emp_log_f_max,importance_orthants,importance_mass=batch_sampling(no_samples=new_batch_size,theta_sampling=theta_sampling,importance_directions=importance_directions,
                                                              importance_orthants=importance_orthants,importance_mass=importance_mass,first=False,old_log_f_max=emperical_log_f_max)
                print(emperical_log_f_max,new_emp_log_f_max)
                if new_emp_log_f_max<=emperical_log_f_max:
                    # print('old f_max')
                    u=new_acc.u
                    den=new_acc.sample_log_density
                    mask= emperical_log_f_max+np.array(u)<np.array(den)
                    rejected.extend(new_acc.filter(~mask))
                    new_acc=new_acc.filter(mask)
                
                    accepted.extend(new_acc)
                else:
                    print('new f_max ---- ',new_emp_log_f_max,emperical_log_f_max)
                    emperical_log_f_max=new_emp_log_f_max
                    u=accepted.u
                    den=accepted.sample_log_density
                    mask= emperical_log_f_max+np.array(u)<np.array(den) 
                    
                    newly_rejected=accepted.filter(~mask) 
                    print(len(newly_rejected.u), 'newly rejected from previous ones.')
                    rejected.extend(accepted.filter(~mask))
                    accepted=accepted.filter(mask)
                    accepted.extend(new_acc)
                    # This new rejected ones that come from accepted will be append in the end.
                rejected.extend(new_reject)
                accepted_count=len(accepted.u)
                pbar.update(accepted_count-previous)
                previous=accepted_count
                # print(len(accepted.theta))
            ans_accepted=list(zip(accepted.theta,accepted.r_batch))
            ans_rejected=list(zip(rejected.theta,rejected.r_batch))
        print('Done!')
        print(importance_mass)
        t_main_end=time.perf_counter()
        print(t_main_end-t_main,'whole','--samples per second--',len(ans_accepted)+len(ans_rejected)/(t_main_end-t_main))
        return ans_accepted,ans_rejected




    def sampling_f_r(self,f_r,batch_size=256):
        """
        The main sampling function utilising the concept of n-sphere Monte
        Carlo technique and rejection sampling.
        Returns:
            Two arrays, accepted samples and rejected samples.
        f_r: gives two things, a density function and f_max value in this one.
        
        Note: r^d-1 is alrady multplied in this f_r function.
        """        
        
        density=f_r[0]
        f_max=f_r[1]
        
        accepted_theta_list,accepted_r_list=[],[]
        rejected_theta_list,rejected_r_list=[],[]
        with tqdm(total=self.k,unit='accepted samples') as pbar:
            accepted_count=0
            previous=0
            while accepted_count<self.k:
                if np.abs(accepted_count-self.k)<batch_size:
                    batch_size=2*np.abs(accepted_count-self.k)            

                theta_batch=self.theta_generation(batch_size)
                R_batch=self.R(theta_batch)
                a_batch,b_batch,total_mass_batch=self.importance_r(density,R_batch,theta_batch)
                sampled_r_batch=np.random.uniform(a_batch,b_batch)
                sampled_f=np.random.uniform(0,f_max,size=batch_size)
                density_vals=density(sampled_r_batch,theta_batch)
                mask= sampled_f <= density_vals
                accepted_theta_list.append(theta_batch[mask])
                accepted_r_list.append(sampled_r_batch[mask])
                rejected_theta_list.append(theta_batch[~mask])
                rejected_r_list.append(sampled_r_batch[~mask])
                accepted_count+=np.sum(mask)
                pbar.update(accepted_count-previous)
                previous=accepted_count
                # print(len(sampled_r_batch[~mask]),len(sampled_r_batch[mask]))
        a_theta=np.concatenate(accepted_theta_list, axis=0)
        a_r=np.concatenate(accepted_r_list, axis=0)
        r_theta=np.concatenate(rejected_theta_list, axis=0)
        r_r=np.concatenate(rejected_r_list, axis=0)
        accepted=np.concatenate([a_theta, a_r[:, None]], axis=1)
        rejceted=np.concatenate([r_theta, r_r[:, None]], axis=1)

        return accepted,rejceted



                
