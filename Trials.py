case0 = ["Case0_FOV1", 'Case0_FOV2', 'Case0_FOV3', 'Case0_FOV4']
case1 = ["Case1_FOV1", 'Case1_FOV2', 'Case1_FOV3', 'Case1_FOV4']
case2 = ["Case2_FOV1", 'Case2_FOV2', 'Case2_FOV3', 'Case2_FOV4']


TRIALS = {'trial1': {'train': [],
                     'test:': []},
          'case_extr': {"train": case1 + ['Case2_FOV1', 'Case2_FOV4'],
                        "test": ['Case2_FOV2', 'Case2_FOV3']},
          
          'single_extr_c2_f4': {"train": ['Case2_FOV1', 'Case2_FOV2', 'Case2_FOV3'],
                                "test": ['Case2_FOV4']},
          
          'single_inter_c2_f3': {"train": ['Case2_FOV1', 'Case2_FOV2', 'Case2_FOV4'],
                                 "test": ['Case2_FOV3']},
          
          'single_inter_c2_f2': {"train": ['Case2_FOV1', 'Case2_FOV3', 'Case2_FOV4'],
                                 "test": ['Case2_FOV2']},
          
          'single_fov2':        {"train": ["Case2_FOV1"],
                                 "test": ["Case2_FOV1"]},
          
          'single_fov1':        {"train": ["Case1_FOV1"],
                                  "test": ["Case1_FOV1"]},
          
          'single_fov0':        {"train": ["Case0_FOV1"],
                                 "test": ["Case0_FOV1"]},
          
          'single_case0':       {"train": case0,
                                 "test": case0},
          
          'single_case1':        {"train": case1,
                                  "test": case1},
          
          'single_case2':       {"train": case2,
                                 "test": case2}
          }

debug_trials = {'debug1': {'train': ['Case1_FOV1_data'],
                           'test': ['Case1_FOV1_data']}
                }

'''
normalization
1. dimensional or non dimensional

3. convert to torch.tensor

'''



'''
validation testing

trial 1: 
    input case1 fov1
    output case1 fov1



'''

'''
i.e. interpolate across 1 boudary condition
train on fov 1, 2, 4 predict fov 3
for one case

train on fov 1, 3, 4(if there), predict on 2
across multiple cases
ex
    train:
    case1: Fovs 1, 3, 4
    case2:, 1, 2, 4
    
    test:
    case1 Fov 2
    case 2 Fov 3
    
i.e. extrapolation

training:
    case 1, 2, 4
    fovs all
test on 
    case 3
    fovs. 

'''

