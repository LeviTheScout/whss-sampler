import numpy as np
from tqdm import tqdm

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
        # "f_max give along with f_r , also doing +0,01 in f_max, 
        # might want to remove that later"
        # THIS CAUSED ISSUE, DONT EVER ADD CONSTANT LIKE THIS.
        
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

        # while len(accepted)<self.k:
        #
        #     theta=self.theta_generation()
        #     R_=self.R(theta)
        #
        #     a,b,total_mass=self.importance_r(density, R_,theta,0.01,0.98)
        #     #print(a,b,R_)
        #     sampled_r=np.random.uniform(a,b)
        #     #sampled_r=np.random.uniform(np.sqrt(self.d)-(5/np.sqrt(2)),np.sqrt(self.d)+(5/np.sqrt(2)))
        #     #sampled_r=np.random.uniform(0,R_)
        #     sampled_f=np.random.uniform(0,f_max)
        #
        #     if sampled_f<=density(sampled_r,theta):
        #         accepted.append((theta,sampled_r))
        #     else:
        #         rejected.append((theta,sampled_r))
        #
        #return accepted,rejected

    


    def sampling_f_r_new(self,density,batch_size=256,alpha=0.1):
        """
        alpha: if we want k samples, we will check k+alpha%k samples. eg: alpha=0.01
        """
        def f_max_along_theta(a_batch,b_batch,theta_batch):
            """
            If at some point, decide to put this function outside - then put density as arguemnt too.
            It will generate the maximum of f_r across the whole batch. Returns only 
            one maximum value among all inputs of the batch.
            """

            return  

        
        possible_samples=[]
        maximums=[]
    
        for i in range(round(self.k/batch_size)+round((self.k/batch_size)*alpha)):
            # taking approximately 

            theta_batch=self.theta_generation(batch_size)
            R_batch=self.R(theta_batch)
            a_batch,b_batch,total_mass_batch=self.importance_r(density,R_batch,theta_batch)
            
            # theta=self.theta_generation(self.d)
            # R_=self.R(self.a,theta)
            #
            # a,b,total_mass=self.importance_r(f_r, R_,theta,0.01,0.98)
            
            local_f_max_batch=f_max_along_theta(a_batch,b_batch,theta_batch)
            sampled_r_batch=np.random.uniform(a_batch,b_batch) #use importance sampling in R , [a,b]
            density_vals=density(sampled_r_batch,theta_batch)
            u_batch=np.random.uniform(0,1,batch_size)
            #this possible_samples might need to be changed for this vectorised form.
            
            possible_samples.extend(zip(theta_batch,u_batch,sampled_r_batch,density_vals))
            #Append only the maximum of the whole batch.
            maximums.append(local_f_max_batch)
        emperical_f_max=np.max(np.array(maximums))
        accepted=[]
        rejected=[]
        
        mask=emperical_f_max* possible_samples[:,1] < possible_samples[:,3]
        accepted.append(possible_samples[mask])
        rejected.append(possible_samples[~mask])



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
        '''
        
        

        while len(accepted)<self.k:
            theta=self.theta_generation(self.d)
            R_=self.R(self.a,theta)
            sampled_r=np.random.uniform(0,R_)  #change to [a,b]
            sampled_f=np.random.uniform(0,emperical_f_max)
            if sampled_f<=(sampled_r**(self.d-1))*density(sampled_r,theta):
                accepted.append((theta,sampled_r))
            else:
                rejected.append((theta,sampled_r))
        return accepted,rejected
                
