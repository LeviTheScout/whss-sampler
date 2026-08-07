import numpy as np
from scipy.optimize import toms748
from tqdm import tqdm
import time
from dataclasses import dataclass

from nsmc_sampling.methods.importance import importance_r_numba

@dataclass
class Samples:
    u: np.ndarray
    theta: np.ndarray
    r_batch: np.ndarray
    log_f_max: np.ndarray
    log_mass: np.ndarray       # <--- CHANGED: Added log_mass property
    sample_log_density: np.ndarray
    
    def extend(self, other):
        self.u = np.concatenate([self.u, other.u], axis=0)
        self.theta = np.concatenate([self.theta, other.theta], axis=0)
        self.log_f_max = np.concatenate([self.log_f_max, other.log_f_max], axis=0)
        self.log_mass = np.concatenate([self.log_mass, other.log_mass], axis=0)  # <--- CHANGED: Added to extend
        self.r_batch = np.concatenate([self.r_batch, other.r_batch], axis=0)
        self.sample_log_density = np.concatenate([self.sample_log_density, other.sample_log_density], axis=0)
        
    def filter(self, mask):
        return Samples(
            u=self.u[mask],
            theta=self.theta[mask],
            r_batch=self.r_batch[mask],
            log_f_max=self.log_f_max[mask],
            log_mass=self.log_mass[mask],  # <--- CHANGED: Added to filter
            sample_log_density=self.sample_log_density[mask]
        ) 

    def length(self):
        return len(self.u)


class sampling:
    
    def _sampling_f_r_new(self,density,batch_size=3256,alpha=0.1,thresh_acceptance=0.1,angle_importance=np.pi/10,tau=0.01):
        rng = np.random.default_rng()
        t_main = time.perf_counter()
        
        def _batch_sampling_uniform(no_samples, first, importance_orthants=None, importance_weights=None, old_log_mass_max=None):
            top_theta, top_orthants, top_weights = [], [], []
            maximums_log = []
            
            # <--- CHANGED: Initialize with empty log_mass array
            possible_samples = Samples(u=np.array([]), theta=np.empty((0,self.d)), r_batch=np.array([]), 
                                       log_f_max=np.array([]), log_mass=np.array([]), sample_log_density=np.array([]))
            
            for _ in range(np.maximum(round(no_samples/batch_size)+round((no_samples/self.k)*alpha),1)):
                theta_batch = self.theta_generation(batch_size)
                R_batch = self.R(theta_batch)
                a_batch, b_batch, log_f_max_batch, total_mass_batch = self.importance_r(density, R_batch, theta_batch)
                
                # <--- CHANGED: Calculate log_width and log_mass
                log_width_batch = np.log(b_batch - a_batch + 1e-10)
                log_mass_batch = log_f_max_batch + log_width_batch

                sampled_r_batch = np.random.uniform(a_batch, b_batch)
                density_vals = density(sampled_r_batch, theta_batch)

                u_batch = np.log(np.random.uniform(0, 1, len(theta_batch)))
                
                # <--- CHANGED: Pass log_mass_batch into Samples
                batch_samples = Samples(u=u_batch, theta=theta_batch, r_batch=sampled_r_batch, 
                                        log_f_max=log_f_max_batch, log_mass=log_mass_batch, sample_log_density=density_vals)
                
                possible_samples.extend(batch_samples)
                
                # <--- CHANGED: Track global max using log_mass_batch
                maximums_log.append(np.max(log_mass_batch))

                if first:
                    weights = log_f_max_batch 
                    top_orthants_batch, top_theta_batch, top_weight_batch = self.away_thetas_batch(theta_batch, weights, tau, batch=True)
                    top_theta.extend(top_theta_batch)
                    top_weights.extend(top_weight_batch)
                    top_orthants.extend(top_orthants_batch)
                else:
                    weights = log_f_max_batch
                    top_orthants_batch, top_theta_batch, top_weight_batch = self.away_thetas_batch(theta_batch, weights, tau, batch=True)
                    
            new_emp_log_mass_max = np.max(np.array(maximums_log))
            
            if old_log_mass_max is not None and new_emp_log_mass_max < old_log_mass_max:
                new_emp_log_mass_max = old_log_mass_max
                    
            # MASK 1: Uses log_f_max (correct for local rejection)
            mask1 = possible_samples.log_f_max + possible_samples.u < possible_samples.sample_log_density
            accepted_temp = possible_samples.filter(mask1)
            rejected = possible_samples.filter(~mask1)

            # <--- CHANGED: MASK 2 uses log_mass (correct for global rejection)
            u2 = np.log(np.random.uniform(0, 1, len(accepted_temp.log_mass)))
            mask2 = u2 < accepted_temp.log_mass - new_emp_log_mass_max
            accepted = accepted_temp.filter(mask2)
            rejected_temp = accepted_temp.filter(~mask2)
            rejected.extend(rejected_temp)
            
            if first:
                top_m_orthants, top_m_theta, corresponding_weights = self.away_thetas_batch(np.array(top_theta), np.array(top_weights), tau, orthants_batch=np.array(top_orthants), batch=False)
                return accepted, rejected, new_emp_log_mass_max, top_m_orthants, top_m_theta, corresponding_weights
            
            return accepted, rejected, new_emp_log_mass_max, importance_orthants, importance_weights
            
        
        def _batch_sampling_orthant(no_samples, importance_orthants, importance_weights):
            accepted = Samples(u=np.array([]), theta=np.empty((0,self.d)), r_batch=np.array([]), 
                               log_f_max=np.array([]), log_mass=np.array([]), sample_log_density=np.array([]))
            rejected = Samples(u=np.array([]), theta=np.empty((0,self.d)), r_batch=np.array([]), 
                               log_f_max=np.array([]), log_mass=np.array([]), sample_log_density=np.array([]))
            
            number_of_orthants = len(importance_orthants)
            
            for _ in range(np.maximum(round(no_samples/batch_size)+round((no_samples/self.k)*alpha),1)):
                
                # 1. Select an Orthant using the PEAK weight
                selected = False
                max_weight = np.max(importance_weights)
                while not selected:
                    index = rng.integers(number_of_orthants) 
                    u1 = np.log(rng.uniform(0, 1))
                    # Probability of picking orthant is proportional to its peak weight
                    selected = u1 <= importance_weights[index] - max_weight
                    
                # 2. Sample inside the chosen orthant
                theta_batch = self.orthant_theta_generator(importance_orthants[index], batch_size)
                R_batch = self.R(theta_batch)
                a_batch, b_batch, log_f_max_batch, total_mass_batch = self.importance_r(density, R_batch, theta_batch)
                
                # We still calculate log_mass_batch so it can be stored in the Samples object
                log_width_batch = np.log(b_batch - a_batch + 1e-10)
                log_mass_batch = log_f_max_batch + log_width_batch
                
                sampled_r_batch = np.random.uniform(a_batch, b_batch)
                density_vals = density(sampled_r_batch, theta_batch) 
                
                # 3. Two-Step Rejection
                u_batch = np.log(np.random.uniform(0, 1, len(theta_batch)))
                
                # Mask 1: Standard local rejection along the ray
                mask1 = u_batch + log_f_max_batch < density_vals
                
                # Mask 2: Reject based on the ray's PEAK vs the ORTHANT's expected PEAK
                u2 = np.log(np.random.uniform(0, 1, len(theta_batch)))
                mask2 = u2 < log_f_max_batch - importance_weights[index]  # FIXED: Now uses log_f_max_batch
                
                # Combine masks
                final_mask = mask1 & mask2
                
                batch_samples = Samples(u=u_batch, theta=theta_batch, r_batch=sampled_r_batch, 
                                        log_f_max=log_f_max_batch, log_mass=log_mass_batch, sample_log_density=density_vals)
                
                accepted.extend(batch_samples.filter(final_mask))
                rejected.extend(batch_samples.filter(~final_mask))
                
                # 4. Update the orthant weight if we found a HIGHER PEAK inside it
                current_max_peak = np.max(log_f_max_batch)  # FIXED: Now strictly tracks max peak
                if current_max_peak > importance_weights[index]:
                    importance_weights[index] = current_max_peak

            return accepted, rejected, importance_orthants, importance_weights

        with tqdm(total=self.k, unit=' accepted samples ', disable=False) as pbar:
            previous = 0
            importance_theta = False
            
            accepted, rejected, emperical_log_mass_max, importance_orthants, top_m_theta, importance_weights = _batch_sampling_uniform(self.k+round(self.k*alpha), first=True)
            
            accepted_count = len(accepted.u)
            pbar.update(accepted.length())
            rejected_count = len(rejected.u)
            
            acceptance_ratio = accepted_count / (accepted_count + rejected_count)
            
            if acceptance_ratio < thresh_acceptance:
                importance_theta = True
                print('switching to importance_orthants!')
                
            while (self.k - accepted_count) > 0:
                remaining = self.k - accepted_count
                new_batch_size = remaining + round(remaining * alpha)
                
                if importance_theta:
                    # FIXED: Removed old_log_mass_max and adjusted return variables
                    new_acc, new_reject, importance_orthants, importance_weights = _batch_sampling_orthant(
                        no_samples=new_batch_size, 
                        importance_orthants=importance_orthants, 
                        importance_weights=importance_weights
                    )
                    
                    # Orthant samples are already unbiased, just add them directly!
                    accepted.extend(new_acc)
                    rejected.extend(new_reject)
                    
                else:
                    new_acc, new_reject, new_emp_log_mass_max, importance_orthants, importance_weights = _batch_sampling_uniform(
                        no_samples=new_batch_size, 
                        first=False,
                        importance_orthants=importance_orthants,
                        importance_weights=importance_weights,
                        old_log_mass_max=emperical_log_mass_max
                    )
                    
                    # Retrospective Pruning safely operates ONLY on uniform samples
                    if new_emp_log_mass_max > emperical_log_mass_max:  
                        u2 = np.log(np.random.uniform(0, 1, len(accepted.u)))
                        
                        mask2 = u2 <= emperical_log_mass_max - new_emp_log_mass_max
                        
                        temp_rejected = accepted.filter(~mask2)
                        rejected.extend(temp_rejected)
                        accepted = accepted.filter(mask2)
                        
                        emperical_log_mass_max = new_emp_log_mass_max
                        
                    # Add new uniform samples AFTER retrospectively pruning old ones
                    accepted.extend(new_acc)
                    rejected.extend(new_reject)
                    
                accepted_count = len(accepted.u)
                pbar.update(accepted_count - previous)
                previous = accepted_count
                
            ans_accepted = list(zip(accepted.theta, accepted.r_batch))
            ans_rejected = list(zip(rejected.theta, rejected.r_batch))
            
        t_main_end = time.perf_counter()
        return ans_accepted, ans_rejected
# import numpy as np
# from scipy.optimize import toms748
# from tqdm import tqdm
# import time
# from dataclasses import dataclass
#
# from nsmc_sampling.methods.importance import importance_r_numba
#
# @dataclass
# class Samples:
#     u: np.ndarray
#     theta: np.ndarray
#     r_batch: np.ndarray
#     log_f_max:np.ndarray
#     sample_log_density: np.ndarray
#
#     def extend(self, other):
#         self.u = np.concatenate([self.u, other.u], axis=0)
#         self.theta = np.concatenate([self.theta, other.theta], axis=0)
#         self.log_f_max = np.concatenate([self.log_f_max, other.log_f_max], axis=0)
#         self.r_batch = np.concatenate([self.r_batch, other.r_batch], axis=0)
#         self.sample_log_density = np.concatenate([self.sample_log_density, other.sample_log_density], axis=0)
#     def filter(self, mask):
#         return Samples(
#             u=self.u[mask],
#             theta=self.theta[mask],
#             r_batch=self.r_batch[mask],
#             log_f_max=self.log_f_max[mask],
#             sample_log_density=self.sample_log_density[mask]
#         ) 
#
#     def length(self):
#         return len(self.u)
#
#
# class sampling:
#
#
#     def _sampling_f_r_new(self,density,batch_size=3256,alpha=0.1,thresh_acceptance=0.1,angle_importance=np.pi/10,tau=0.01):
#         rng=np.random.default_rng()
#         t_main=time.perf_counter()
#
#         # batch_size=self.k+200
#
#         def _batch_sampling_uniform(no_samples, first, importance_orthants= None, importance_weights= None, old_log_f_max= None):
#             top_theta,top_orthants,top_weights=[],[],[]
#             maximums_log=[]
#             possible_samples=Samples(u=np.array([]),theta=np.empty((0,self.d)),r_batch=np.array([]),log_f_max=np.array([]),sample_log_density=np.array([]))
#
#             for _ in range(np.maximum(round(no_samples/batch_size)+round((no_samples/self.k)*alpha),1)):
#                 theta_batch=self.theta_generation(batch_size)
#                 R_batch=self.R(theta_batch)
#                 a_batch,b_batch,log_f_max_batch,total_mass_batch=self.importance_r(density,R_batch,theta_batch)
#                 # u = np.random.uniform(0, 1, len(a_batch))
#                 # sampled_r_batch= R_batch * (u**(1/self.d))
#                 sampled_r_batch=np.random.uniform(a_batch,b_batch)
#                 density_vals=density(sampled_r_batch,theta_batch)
#
#
#                 u_batch=np.log(np.random.uniform(0,1,len(theta_batch)))
#                 batch_samples=Samples(u=u_batch,theta=theta_batch,r_batch=sampled_r_batch,log_f_max=log_f_max_batch,sample_log_density=density_vals)
#                 possible_samples.extend(batch_samples)
#                 maximums_log.append(np.max(log_f_max_batch))
#
#                 if first:
#                     weights= log_f_max_batch # total_mass_batch
#                     top_orthants_batch,top_theta_batch,top_weight_batch=self.away_thetas_batch(theta_batch,weights,tau,batch=True)
#                     top_theta.extend(top_theta_batch)
#                     top_weights.extend(top_weight_batch)
#                     top_orthants.extend(top_orthants_batch)
#                 else:
#                     weights=log_f_max_batch
#                     top_orthants_batch,top_theta_batch,top_weight_batch=self.away_thetas_batch(theta_batch,weights,tau,batch=True)
#                     # see any new orthants that we encountered , if so add them to importance_orthants and with importance_weights
#
#                     # for already existing orthants if we find corresponding_weights higher, then update them in importance_weights
#
#             new_emp_log_f_max=np.max(np.array(maximums_log))
#
#             if old_log_f_max is not None and new_emp_log_f_max < old_log_f_max:
#                 new_emp_log_f_max=old_log_f_max
#
#             mask1 = possible_samples.log_f_max+possible_samples.u < possible_samples.sample_log_density
#             accepted_temp=possible_samples.filter(mask1)
#             rejected=possible_samples.filter(~mask1)
#             print(accepted_temp.length(), rejected.length())
#
#
#             u2=np.log(np.random.uniform(0,1,len(accepted_temp.log_f_max)))
#             mask2= u2 < accepted_temp.log_f_max - new_emp_log_f_max
#             accepted=accepted_temp.filter(mask2)
#             rejected_temp=accepted_temp.filter(~mask2)
#             rejected.extend(rejected_temp)
#             print(accepted.length(), rejected_temp.length())
#
#
#
#
#             # change accepted_temp to accepted
#             # change accepted_temp to accepted
#             # change accepted_temp to accepted
#             # change accepted_temp to accepted
#             # change accepted_temp to accepted
#             if first:
#                 top_m_orthants,top_m_theta, corresponding_weights=self.away_thetas_batch(np.array(top_theta),np.array(top_weights),tau,orthants_batch=np.array(top_orthants),batch=False)
#                 return accepted,rejected,new_emp_log_f_max,top_m_orthants,top_m_theta,corresponding_weights
#             return accepted, rejected,new_emp_log_f_max, importance_orthants, importance_weights
#
#
#
#         def _batch_sampling_orthant(no_samples, importance_orthants, importance_weights, old_log_f_max):
#             maximums_log=[]
#             accepted=Samples(u=np.array([]),theta=np.empty((0,self.d)),r_batch=np.array([]),log_f_max=np.array([]),sample_log_density=np.array([]))
#
#             rejected=Samples(u=np.array([]),theta=np.empty((0,self.d)),r_batch=np.array([]),log_f_max=np.array([]),sample_log_density=np.array([]))
#             for _ in range(np.maximum(round(no_samples/batch_size)+round((no_samples/self.k)*alpha),1)):
#
#                 selected=False
#                 number_of_orthants=len(importance_orthants)
#                 max_weight=np.max(importance_weights)
#                 # print(max_weight, old_log_f_max , 'should be equal.')
#                 while not selected:
#                     index=rng.integers(number_of_orthants-1)
#                     #here weight: log_f_max is in log space.
#                     u1=np.log(rng.uniform(0,1))
#                     selected= u1 <=  importance_weights[index] - max_weight
#                     if selected:
#                         # print(np.unpackbits(importance_orthants[index]),max_weight+u1, importance_weights[index])
#                         theta_batch=self.orthant_theta_generator(importance_orthants[index],batch_size)
#                         R_batch=self.R(theta_batch)
#                         a_batch,b_batch,log_f_max_batch,total_mass_batch=self.importance_r(density,R_batch,theta_batch)
#                         # u = np.random.uniform(0, 1, len(a_batch))
#                         # sampled_r_batch = (a_batch**self.d + u * (b_batch**self.d - a_batch**self.d))**(1/self.d)
#
#                         log_width = np.log(b_batch - a_batch + 1e-10)
#                         log_mass_batch = log_f_max_batch + log_width
#
#                         sampled_r_batch=np.random.uniform(a_batch,b_batch)
#                         density_vals=density(sampled_r_batch,theta_batch)-importance_weights[index]
#                         # print(np.unpackbits(importance_orthants[index])) 
#                         # change here if we change weight to something else.
#                         # print(density_vals+importance_weights[index])                        
#                         current_orthant_weight=np.max(log_f_max_batch)
#                         if current_orthant_weight>=importance_weights[index]:
#                             importance_weights[index]=current_orthant_weight
#
#                         #dividing density with mass of that orthant, aka the bias mitigation step.
#                 u_batch=np.log(np.random.uniform(0,1,len(theta_batch)))
#                 batch_samples=Samples(u=u_batch,theta=theta_batch,r_batch=sampled_r_batch,log_f_max=log_f_max_batch,sample_log_density=density_vals)
#
#                 mask= batch_samples.u <= batch_samples.sample_log_density
#                 accepted.extend(batch_samples.filter(mask))
#                 rejected.extend(batch_samples.filter(~mask))
#                 # print(rejected.length(),accepted.length()) 
#                 maximums_log.append(np.max(log_mass_batch))
#
#
#             new_emp_log_f_max=np.max(np.array(maximums_log))
#             if old_log_f_max is not None and new_emp_log_f_max < old_log_f_max:
#                 new_emp_log_f_max=old_log_f_max
#             # i might encounter new f_max for each batch -- its orthant and it will be updated.
#             # with that weights the overall f_max also changes.
#             return accepted,rejected, new_emp_log_f_max, importance_orthants, importance_weights
#
#
#         with tqdm(total=self.k,unit=' accepted samples ',disable=False) as pbar:
#             previous=0
#             importance_theta=False
#             accepted,rejected,emperical_log_f_max, importance_orthants,top_m_theta, importance_weights=_batch_sampling_uniform(self.k+round(self.k*alpha), first=True)
#             accepted_count=len(accepted.u)
#             pbar.update(accepted.length())
#             rejected_count=len(rejected.u)
#             acceptance_ratio=accepted_count/(accepted_count+rejected_count)
#             # print(acceptance_ratio,'acceptance_ratio',thresh_acceptance)
#             if acceptance_ratio < thresh_acceptance:
#                 importance_theta=True
#                 print('switching to importance_orthants!')
#             while (self.k-accepted_count)>0:
#
#                 remaining=self.k-accepted_count
#                 new_batch_size=remaining+round(remaining*alpha)
#                 if importance_theta:
#                     new_acc,new_reject,new_emp_log_f_max,importance_orthants,importance_weights=_batch_sampling_orthant(no_samples=new_batch_size,
#                                                               importance_orthants=importance_orthants,importance_weights=importance_weights,old_log_f_max=emperical_log_f_max)
#                 else:
#                     new_acc,new_reject,new_emp_log_f_max,importance_orthants,importance_weights=_batch_sampling_uniform(no_samples=new_batch_size,first=False
#                                                               ,importance_orthants=importance_orthants,importance_weights=importance_weights,old_log_f_max=emperical_log_f_max)
#
#                 # print(new_emp_log_f_max, emperical_log_f_max)
#                 if new_emp_log_f_max > emperical_log_f_max:  # = or != ?
#                     u2= np.log(np.random.uniform(0,1,len(accepted.u)))
#                     mask2= u2 <= emperical_log_f_max - new_emp_log_f_max
#                     temp_rejcted=accepted.filter(~mask2)
#                     rejected.extend(temp_rejcted)
#                     accepted=accepted.filter(mask2)
#                     # print('new f_max', new_emp_log_f_max, 'new rejected', temp_rejcted.length())
#                     emperical_log_f_max=new_emp_log_f_max
#                 # we are already taking care of the 'else' case inside batch_sampling function/
#                 # i.e. new_emp_log_f_max < emperical_log_f_max --- old samples are fine, new samples in batch_Sampling will be taken care using the old_emperical_log_f_max.
#                 accepted.extend(new_acc)
#                 rejected.extend(new_reject)
#
#
#                 accepted_count=len(accepted.u)
#                 pbar.update(accepted_count-previous)
#                 previous=accepted_count
#                 # print(len(accepted.theta))
#             ans_accepted=list(zip(accepted.theta,accepted.r_batch))
#             ans_rejected=list(zip(rejected.theta,rejected.r_batch))
#         # print('Done!')
#         t_main_end=time.perf_counter()
#         # print(t_main_end-t_main,'whole','--samples per second--',len(ans_accepted)+len(ans_rejected)/(t_main_end-t_main))
#         return ans_accepted,ans_rejected
#
#         return t_main_end-t_main, len(ans_rejected)

    def sampling_f_r_new_old(self,density,batch_size=256,alpha=0.1,thresh_acceptance=0.1,angle_importance=np.pi/10,tau=0.01):

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
            possible_samples=Samples(u=np.array([]),theta=np.empty((0,self.d)),r_batch=np.array([]),log_f_max=np.array([]),sample_log_density=np.array([]))
            for _ in range(np.maximum(round(no_samples/batch_size)+round((no_samples/self.k)*alpha),1)):
                if theta_sampling:
                    # will do rejection sampling on orthants to choose one for each batch.
                    selected=False
                    number_of_orthants=len(importance_orthants)
                    max_mass=np.max(importance_mass)
                    while not selected:
                        index=rng.integers(number_of_orthants-1)
                        selected= max_mass*(rng.uniform(0,1))<=importance_mass[index]
                        if selected:
                            theta_batch=self.orthant_theta_generator(importance_orthants[index],batch_size)
                            R_batch=self.R(theta_batch)
                            a_batch,b_batch,log_f_max_batch,total_mass_batch=self.importance_r(density,R_batch,theta_batch)
                            sampled_r_batch=np.random.uniform(a_batch,b_batch)
                            density_vals=density(sampled_r_batch,theta_batch)-np.log(importance_mass[index])
                            
                            new_orthant_mass=np.max(log_f_max_batch)
                            if new_orthant_mass>=importance_mass[index]:
                                importance_mass[index]=new_orthant_mass

                            #dividing density with mass of that orthant, aka the bias mitigation step.
                else:
                    theta_batch=self.theta_generation(batch_size)
                    R_batch=self.R(theta_batch)
                    a_batch,b_batch,log_f_max_batch,total_mass_batch=self.importance_r(density,R_batch,theta_batch)
                #use this total_mass_batch for the running mean and variance calculation.
                    sampled_r_batch=np.random.uniform(a_batch,b_batch)
                    density_vals=density(sampled_r_batch,theta_batch)
                    
                u_batch=np.log(np.random.uniform(0,1,len(theta_batch)))
                batch_samples=Samples(u=u_batch,theta=theta_batch,r_batch=sampled_r_batch,log_f_max=log_f_max_batch,sample_log_density=density_vals)
                possible_samples.extend(batch_samples)
                maximums_log.append(np.max(log_f_max_batch))
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
            
            emperical_log_f_max=np.max(np.array(maximums_log))
            
            if old_log_f_max is not None and emperical_log_f_max <= old_log_f_max:
                print(np.max(np.array(maximums_log)))
                emperical_log_f_max=old_log_f_max
                
            if theta_sampling:
               mask= possible_samples.u<possible_samples.sample_log_density
               accepted=possible_samples.filter(mask)
               rejected=possible_samples.filter(~mask)
            else:
                
                mask1 = possible_samples.log_f_max+possible_samples.u<possible_samples.sample_log_density
                accepted_temp=possible_samples.filter(mask1)
                rejected=possible_samples.filter(~mask1)
                u2=np.log(np.random.uniform(0,1,len(accepted_temp.log_f_max)))
                mask2=u2<accepted_temp.log_f_max - emperical_log_f_max
                print(np.sum(mask2),len(accepted_temp.u),np.sum(mask1),len(rejected.u),len(possible_samples.u),'mask2')     
                accepted=accepted_temp.filter(mask2)
                rejected_temp=accepted_temp.filter(~mask2)

                rejected.extend(rejected_temp)
             
            # will have to check again for away directions before adding to the global list.
            if first:
                # change tau here. 
                # tau = factor / mean, need to make sure it is fine for both batch wise and global.
                top_m_orthants,top_m_theta, corresponding_masses=self.away_thetas_batch(np.array(top_mass_theta),np.array(top_masses),thresh_angle,tau,global_mean=running_mean,orthants_batch=np.array(top_orthants),batch=False)
                # print(len(top_m_orthants))
                print('function end')
                return accepted,rejected,emperical_log_f_max,top_m_orthants,top_m_theta,corresponding_masses
            
            return accepted,rejected,emperical_log_f_max,importance_orthants,importance_mass


        with tqdm(total=self.k,unit=' accepted samples ') as pbar:
            previous=0

            accepted,rejected,emperical_log_f_max, top_m_orthants,top_m_theta, top_masses=batch_sampling(self.k+round(self.k*alpha),first=True)
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



                
