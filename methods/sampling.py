from typing import dataclass_transform
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class Samples:
    u: np.ndarray
    theta: np.ndarray
    r_batch: np.ndarray
    density: np.ndarray
    
    def extend(self, other):
        self.u = np.concatenate([self.u, other.u], axis=0)
        self.theta = np.concatenate([self.theta, other.theta], axis=0)
        self.r_batch = np.concatenate([self.r_batch, other.r_batch], axis=0)
        self.density = np.concatenate([self.density, other.density], axis=0)
    def filter(self, mask):
        return Samples(
            u=self.u[mask],
            theta=self.theta[mask],
            r_batch=self.r_batch[mask],
            density=self.density[mask]
        )   



class sampling:

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



    def sampling_f_r_new(self,density,thresh_angle=np.pi/6,batch_size=256,alpha=0.1,m=10):
        """
        alpha: if we want k samples, we will check k+alpha%k samples. eg: alpha=0.01
        batch_size: no. of samples procced at a time.
        m: no. of directions to open cone for importance sampling.
        """

        maximums=[]
        away_top_mass_directions=[]
        def batch_sampling(no_samples):   
            possible_samples=Samples(u=np.array([]),theta=np.empty((0,self.d)),r_batch=np.array([]),density=np.array([]))
            for i in range(np.maximum(round(no_samples/batch_size)+round((no_samples/self.k)*alpha),1)):
                theta_batch=self.theta_generation(batch_size)
                R_batch=self.R(theta_batch)
                a_batch,b_batch,local_f_max_batch,total_mass_batch=self.importance_r(density,R_batch,theta_batch)
                sampled_r_batch=np.random.uniform(a_batch,b_batch) #use importance sampling in R , [a,b]
                density_vals=density(sampled_r_batch,theta_batch)
                u_batch=np.random.uniform(0,1,batch_size)
                batch_samples=Samples(u=u_batch,theta=theta_batch,r_batch=sampled_r_batch,density=density_vals)
                possible_samples.extend(batch_samples)
                maximums.append(np.max(local_f_max_batch))
                
                #for importance in theta
                
                top_m_theta_batch=self.away_thetas_batch(theta_batch,total_mass_batch,thresh_angle,m)
                #check if they are far enough, if not, discard smaller masses.
                # this benifits me since i would like to keep list as small as possible.
                # this also makes sure my saved directions are separate enough and large enoguh for 
                #each batch. as batch gets added, it should check again with previos batch masses.
                # basically we maintain far enough directions and then eventully just sort and select
                # required number of directions based on mass.
            
            # will have to check again for away directions before adding to the global list.
            emperical_f_max=np.max(np.array(maximums))
            mask = emperical_f_max * possible_samples.u < possible_samples.density
            accepted=possible_samples.filter(mask)
            rejected=possible_samples.filter(~mask)
            # print(len(accepted.u),len(rejected.u)) 
            return accepted,rejected,emperical_f_max 
                
        with tqdm(total=self.k,unit=' accepted samples ') as pbar:
            previous=0
            accepted,rejected,emperical_f_max=batch_sampling(self.k+round(self.k*alpha))
            accepted_count=len(accepted.u)
            while (self.k-accepted_count)>0:
                remaining=self.k-accepted_count
                new_acc,new_reject,new_emp_max=batch_sampling(remaining+round(remaining*alpha))
                # we put condtion here for importance theta, if i have acceptance ratio
                # smaller than 'threshold' then will go for importance theta sampling.
                if new_emp_max<=emperical_f_max:
                    accepted.extend(new_acc)
                else:
                    u=accepted.u
                    den=accepted.density
                    mask= emperical_f_max*np.array(u)<np.array(den)
                    
                    accepted=accepted.filter(mask)
                    rejected.extend(accepted.filter(~mask))
                    accepted.extend(new_acc)
                    # This new rejected ones that come from accepted will be append in the end.
                rejected.extend(new_reject)
                accepted_count=len(accepted.u)
                pbar.update(accepted_count-previous)
                previous=accepted_count
            ans_accepted=list(zip(accepted.theta,accepted.r_batch))
            ans_rejected=list(zip(rejected.theta,rejected.r_batch))
        return ans_accepted,ans_rejected
                
        '''
        if i get less than k samples, then i will do normal rejection sampling 
        with emperical f_max for remaining samples, i will deploy the importance sampling in r, 
        hence it should not come to this case more often.

        - NO. Sir's idea is to go for more sample same as previous loop, and if I find
        the emperical_f_max higher than previous one, then we do retrospective prunning. 
        
        - Will do this newer sample search for some number with respect to remaining numbner
        of samples. something like --- 
        (remaining_no_samples)+(remaining_no_samples)*alpha
        
        - Might work, since there is less chance of finding bigger_f_max. if i do then 
        i will only have to see ONLY previous accepted samples (not rejected ones since they are 
        even smaller.)

        - this surely will increase rejection ratio?
        '''
