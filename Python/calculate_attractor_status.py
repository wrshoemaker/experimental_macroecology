


import utils


experiments = [('No_migration', 4), ('No_migration', 40), ('Parent_migration', 4), ('Global_migration', 4)]


for experiment in experiments:


    attractor_dict = utils.get_attractor_status(migration=experiment[0], inocula=experiment[1])

    if 'Alcaligenaceae' in attractor_dict:
        n_alcaligenaceae = len(attractor_dict['Alcaligenaceae'])

    else:
        n_alcaligenaceae = 0

    if 'Pseudomonadaceae' in attractor_dict:
        n_pseudomonadaceae = len(attractor_dict['Pseudomonadaceae'])
    else:
        n_pseudomonadaceae = 0

    f_alcaligenaceae = n_alcaligenaceae/(n_alcaligenaceae+n_pseudomonadaceae)

    print(experiment, f_alcaligenaceae)
