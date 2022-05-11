import numpy as np
rng = np.random.default_rng(seed=0)
import concurrent.futures
def simulate_drift_batch(input):
    biallelic_freqs, generation, Ne, Ncensus, Nsampling=input
    Ne=max(1,Ne)
    Ne*=2 # diploid org.
    new_freqs=np.asarray(biallelic_freqs)
    q=1.0-new_freqs
    n=len(biallelic_freqs)
    for _ in range(generation):
        new_freqs=rng.binomial(Ne,new_freqs,n)/Ne

    if Nsampling<Ncensus:
        new_freqs=rng.hypergeometric(ngood=new_freqs*Ncensus*2,nbad=q*Ncensus*2,nsample=Nsampling*2,size=n)

    return new_freqs

def create_batch_biallelic_freqs(g0):
    biallelic_freqs, positions_major,positions_minor,del_positions=[],[],[],[]
    tmp=np.logical_and((g0!=1),(g0!=0))
    positions_tmp=np.where(tmp)
    major_freqs=[]

    positions_tmp=sorted(list(set(positions_tmp[0])))
    #search position with highest and second highest as major and minor + recalculate freq?!
    for pos in positions_tmp:
        indices = np.argsort(g0[pos])
        if g0[pos,indices[1]]:
            new_freq_major=float(g0[pos,indices[3]]/(g0[pos,indices[2]]+g0[pos,indices[3]]))
            del_positions.append([pos,indices[1]])
            if g0[pos,indices[0]]:
                del_positions.append([pos, indices[0]])
        else:
            new_freq_major=float(g0[pos,indices[3]])
        #new_freq_major=round(new_freq_major,3)
        major_freqs.append(new_freq_major)
        biallelic_freqs.append(new_freq_major)#+=str(new_freq_major)+':'
        positions_major.append(indices[3])
        positions_minor.append(indices[2])
    row_pos=positions_tmp
    #[:-1]
    return biallelic_freqs,positions_major,positions_minor,del_positions,row_pos,np.asarray(major_freqs)


def batch_simulation_creation(g0,gn,Ne,Ncensus,Nsampling,generation,cam_data):
    #Idea 1: for every batch item make a thread  executing
    num_cams, variation=cam_data
    biallelic_freqs, positions_major, positions_minor, del_positions,rows,major_freqs=create_batch_biallelic_freqs(g0)
    Nes=[]
    if len(positions_major):

        if len(del_positions):
            rows_,cols=zip(*del_positions)
            g0[rows_,cols]=[0]*len(del_positions)
            gn[ rows_, cols] = [0] * len(del_positions)

        g0 = np.repeat(g0[np.newaxis, :, :], num_cams, axis=0)
        gn = np.repeat(gn[np.newaxis, :, :], num_cams, axis=0)
        gN = np.copy(g0)
        Nes=rng.normal(Ne,variation,num_cams)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(simulate_drift_batch, (biallelic_freqs,generation,Nes[i],Ncensus,Nsampling ))   for i in range(num_cams)]
        gN_freqs=[f.result() for f in futures]

        for i in range(num_cams):
            gN[i,rows, positions_major] = gN_freqs[i]
            gN_minor_freqs=1-gN_freqs[i]
            gN[i,rows, positions_minor] = gN_minor_freqs
            g0[i,rows, positions_major] = major_freqs
            g0_minor_freqs=1-major_freqs
            g0[i,rows, positions_minor] = g0_minor_freqs

        #fake_g_,new_g0_,new_gn_
        return gN,g0,gn,Nes,len(positions_major)#gN,g0,gn,Nes,len(positions_major)#g0,gN,gn,Nes,len(positions_major)#
    g0 = np.repeat(g0[np.newaxis, :, :], num_cams, axis=0)
    gn = np.repeat(gn[np.newaxis, :, :], num_cams, axis=0)
    return g0,g0,gn,Nes,len(positions_major)