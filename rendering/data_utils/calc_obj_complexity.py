import torch
import numpy as np
import pickle



def flow_by_coordinate_matching(
    coord1: torch.Tensor,
    coord2: torch.Tensor,
    index=None,
    train_index: bool=True,
    threshold: float=5e-5
):
    '''Compute ground-truth flow by finding geometrically matching points between two sets of
    surface coordinates.
    '''
    if index is None:
        # NOTE: constructing the index adds significant overhead, it's better to construct it
        # outside and pass it in if calling this function frequently
        import faiss
        import faiss.contrib.torch_utils

        gres = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gres, 0, faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, 32))
        index.nprobe = 2
    
    mask1 = coord1.ne(0).any(-1)
    mask2 = coord2.ne(0).any(-1)

    if train_index:
        index.train(coord1[mask1])
    
    b, h, w = coord1.shape[:-1]
    flow = coord1.new_full((b, 2, h, w), torch.inf)
    
    for i in range(coord1.shape[0]):
        m1, m2 = mask1[i], mask2[i]
        c1, c2 = coord1[i][m1], coord2[i][m2] 
        index.reset()
        index.add(c1)

        p1, p2 = _match_points(c2, m1, m2, index, threshold)
        offsets = p1.sub(p2).float()

        # putting coordinates in as (x, y)
        flow[i, 0].index_put_((p2[:, 0], p2[:, 1]), offsets[:, 1])
        flow[i, 1].index_put_((p2[:, 0], p2[:, 1]), offsets[:, 0])

    return flow




# SEE https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV0405/MUIR/av-2.html
def calc_curvature(
    coords:  torch.Tensor,
    normals: torch.Tensor
):
    Dx = coords[2:,:,:] - coords[:-2,:,:]
    Lx = torch.sqrt(torch.sum(Dx**2,axis=2))
    N1,N2 = normals[:-2,:,:], normals[2:,:,:]
    th = torch.acos(torch.sum(N1*N2,axis=2))
    Kx = 2.*torch.sin(th/2.)/Lx

    Dy = coords[:,2:,:] - coords[:,:-2,:]
    Ly = torch.sqrt(torch.sum(Dy**2,axis=2))
    N1,N2 = normals[:,:-2,:], normals[:,2:,:]
    th = torch.acos(torch.sum(N1*N2,axis=2))
    Ky = 2.*torch.sin(th/2.)/Ly

    
    mean_curv  = (Kx[:,1:-1]+Ky[1:-1,:])/2.
    gauss_curv = (Kx[:,1:-1]*Ky[1:-1,:])

    obj_mask = torch.logical_and(torch.logical_not(torch.isinf(mean_curv)),torch.logical_not(torch.isnan(mean_curv)))

    stats_mean_curv  = (mean_curv[obj_mask].mean(), mean_curv[obj_mask].std())
    stats_gauss_curv = (gauss_curv[obj_mask].mean(), gauss_curv[obj_mask].std())

    return stats_mean_curv, stats_gauss_curv, mean_curv[obj_mask].numel()



PATH = '/nobackup/scratch/grp/grp_farrell/data/synthetic'
ds = 'julia3d'
#ds = 'blob-10k'
#ds = 'cloth-10k'
#ds = 'ifs3d/depth_3'
per_obj_data = []
full_data = []
for obj_num in range(100):
    #obj_num = 0
    obj = f'c_{obj_num:05d}'
    C = np.load(f'{PATH}/{ds}/{obj}/coords.npz')
    N = np.load(f'{PATH}/{ds}/{obj}/normals.npz')

    per_image_data = []
    for img_num in range(100):
        img = f'{img_num:04d}'
        #img = f'{img_num:03d}'
        C_img = torch.Tensor(C[img]).cuda()
        N_img = torch.Tensor(N[img]).cuda()
        tup = ( obj, img, *calc_curvature(C_img,N_img) )
        per_image_data.append( tup )
        #full_data.append( tup )

    
    valid_mask = torch.logical_not( torch.isnan( torch.Tensor( [a[2][0] for a in per_image_data] ) ) )
    mean_curv  = torch.Tensor( [a[2] for a in per_image_data] )
    gauss_curv = torch.Tensor( [a[3] for a in per_image_data] )

    obj_tup = ( obj, mean_curv[valid_mask].mean(axis=0).cpu().numpy(), gauss_curv[valid_mask].mean(axis=0).cpu().numpy() )

    tup_objs  = [a[0] for a in per_image_data]
    tup_objs  = [tup_objs[i] for i in range(len(tup_objs)) if valid_mask[i]]
    tup_imgs  = [a[1] for a in per_image_data]
    tup_imgs  = [tup_imgs[i] for i in range(len(tup_imgs)) if valid_mask[i]]
    mean_pairs = list(zip(mean_curv[valid_mask,0].cpu().numpy(),mean_curv[valid_mask,1].cpu().numpy()))
    #print(mean_pairs)
    gauss_pairs = list(zip(gauss_curv[valid_mask,0].cpu().numpy(),gauss_curv[valid_mask,1].cpu().numpy()))
    tup_sizes = [a[4] for a in per_image_data]
    tup_sizes = [tup_sizes[i] for i in range(len(tup_sizes)) if valid_mask[i]]
    allimgs = list(zip(tup_objs,tup_imgs,mean_pairs,gauss_pairs,tup_sizes))
    #print(allimgs)
    full_data.extend( allimgs.copy() )#per_image_data)
    per_obj_data.append( obj_tup )
    print(obj)

print(full_data[0])
pickle.dump( per_obj_data, open( f'{PATH}/{ds}/curvature.pkl','wb') )
pickle.dump( full_data,    open( f'{PATH}/{ds}/curvature_allimages.pkl','wb') )




    

    


