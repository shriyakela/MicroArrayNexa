import torch
path = 'unet_trained_model.pkl'
opts = [ {}, {'map_location': torch.device('cpu')}, {'map_location': torch.device('cpu'), 'weights_only': True} ]
for kw in opts:
    try:
        print('trying', kw)
        m = torch.load(path, **kw)
        print('loaded ok, type:', type(m))
        break
    except Exception as e:
        print('failed:', e)
