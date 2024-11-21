import MDAnalysis as mda
import numpy as np
import torch
from MDAnalysis import transformations
from mlcolvar.data import DictDataset
from mlcolvar.utils.io import create_dataset_from_files
from ovito.io import import_file
from ovito.modifiers import UnwrapTrajectoriesModifier
from torch_geometric.data import Data, InMemoryDataset

from .utils import compute_committor_weights_wo_dataset


########################################################################
# Base class to create torch geometric dataset from atomistic data
########################################################################
class BaseAtomisticDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def get_atomistic_data(self):
        raise InterruptedError("self.get_atomistic_data needs to be implemented.")

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        X, node_attr, labels, weights, node_inputs = self.get_atomistic_data()
        nsnapshot = len(X)

        for n in range(nsnapshot):

            graph = Data(
                pos=X[n],
                node_attr=node_attr[n],
                labels=labels[n],
                weights=weights[n],
                node_input=node_inputs[n],
            )

            data_list.append(graph)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


########################################################################
# Parse atomistic data for different format
########################################################################


class ExampleDataset(BaseAtomisticDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    def get_atomistic_data(self):

        # 50 snapshots of 30 x y z atoms
        nsapshots = 50
        natoms = 30

        # Get positions and not attr, each graph will represent the whole snapshot
        X = torch.randn((nsapshots, natoms, 3), dtype=torch.get_default_dtype())

        node_attr = torch.tensor(
            [[[1, 0] for _ in range(natoms)] for _ in range(nsapshots)],
            dtype=torch.get_default_dtype(),
        )

        # Half are from bassin A, half from B
        labels = torch.arange((nsapshots))
        # labels[nsapshots // 2 :] = 1

        # !!! make sure to change based on bias beta tec
        bias = torch.zeros(nsapshots)
        data_groups = torch.zeros(
            nsapshots
        )  # Iteration number? Pr really just the labels?
        data_groups[nsapshots // 2 :] = 1

        weights, labels = compute_committor_weights_wo_dataset(
            labels, bias=bias, data_groups=data_groups, beta=1
        )

        return X, node_attr, labels, weights


class CommitorPaperDataset(BaseAtomisticDataset):
    def __init__(
        self,
        root,
        filenames,
        load_args,
        group,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.filenames = filenames
        self.load_args = load_args
        self.group = group
        super().__init__(root, transform, pre_transform, pre_filter)

    def get_atomistic_data(self):
        from mlcolvar.cvs.committor.utils import compute_committor_weights
        from mlcolvar.utils.io import create_dataset_from_files

        # temperature
        T = 1
        # Boltzmann factor in the RIGHT ENREGY UNITS!
        kb = 1
        beta = 1 / (kb * T)

        folder = None

        dataset, dataframe = create_dataset_from_files(
            file_names=self.filenames,
            folder=folder,
            create_labels=True,
            filter_args={
                "regex": "p.x|p.y"
            },  # to load many positions --> 'regex': 'p[1-9]\.[abc]|p[1-2][0-9]\.[abc]'
            return_dataframe=True,
            load_args=self.load_args,
            verbose=True,
        )
        # fill empty entries from unbiased simulations

        if "bias" in dataframe:
            dataframe = dataframe.fillna({"bias": 0})
        else:
            print("Check me. Bias set to zero")

            dataframe["bias"] = 0

        bias = torch.Tensor(dataframe["bias"].values)

        dataset = compute_committor_weights(dataset, bias, self.group, beta)

        labels = dataset["labels"]
        weights = dataset["weights"]

        X = dataset["data"]

        # Doesn't work if only one node because edge_vec are zeros

        #!!! ADD BACK FOR 3x3 output
        pos = torch.zeros((X.shape[0], X.shape[1] + 1))
        pos[:, 0] = X[:, 0]
        pos[:, 1] = X[:, 1]
        X = pos
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        # for i in range(len(X)):

        #     X[i]= torch.cat([X[i], torch.tensor([[1,1,1]])])

        # add = torch.ones((len(X), 1, 3), dtype=torch.get_default_dtype())  #!!!
        # X = torch.concat([X, add],dim=1)
        # X = add.reshape((4000, 2, 3))

        node_attr = torch.tensor(
            [[[1, 0] for _ in range(len(X[0]))] for _ in range(len(X))],
            dtype=torch.get_default_dtype(),
        )
        node_inputs = X  # torch.ones(X.shape)

        return X, node_attr, labels, weights, node_inputs


class GromacsDataset(BaseAtomisticDataset):
    def __init__(
        self,
        root,
        filenames,
        load_args,
        group,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.filenames = filenames
        self.load_args = load_args
        self.group = group
        super().__init__(root, transform, pre_transform, pre_filter)

    def get_atomistic_data(self):

        # X = []
        # for filename in self.filenames:
        #     print(f'Doing {filename}...')

        #     # u = mda.Universe(self.load_args ,filename,guess_bonds=True)
        #     # atoms = u.select_atoms('all')

        #     #unwrapping coordinates
        #     # transform = transformations.unwrap(atoms)
        #     # u.trajectory.add_transformations(transform)
        #     # for ts in range(len(u.trajectory)):
        #     #     X.append(u.trajectory[ts].positions)

        # X = torch.tensor(X, dtype=torch.float64)
        # X = torch.concat(X)

        # node_inputs = torch.ones(X.shape)
        # labels = torch.ones(len(X))
        # labels[:int(len(X)/2)]=0.0
        # weights = torch.ones(len(X))

        # atom_labels = atoms.types
        # unique_types, inverses = np.unique(atom_labels, return_inverse=True)
        # one_hots = np.eye(len(unique_types))[inverses]
        # node_attr = torch.tensor([[one_hots] * len(X)])[0]

        X_AB = []

        for i, filename in enumerate(self.filenames):

            # Getting values for only one dump file
            pipeline = import_file(filename)
            pipeline.modifiers.append(
                UnwrapTrajectoriesModifier()
            )  #!!! ADD back but need to retrain model
            data = pipeline.compute()

            timeframes = (
                range(pipeline.source.num_frames)
                if pipeline.source.num_frames > 1
                else range(1)
            )

            X = []
            for timestep in timeframes:
                data = pipeline.compute(timestep)
                xyz = list(data.particles.positions + 0)

                X.append(np.array(xyz))

            X_AB.append(torch.tensor(X))

        X = torch.concat(X_AB)

        # TODO: atom types might be wrong if OVITO atom ordering is not the same as MDA atom ordering. Let's not forgot to check this. 
        u = mda.Universe(self.load_args, filename, guess_bonds=True)
        atoms = u.select_atoms("all")
        atom_labels = atoms.types
        unique_types, inverses = np.unique(atom_labels, return_inverse=True)
        one_hots = np.eye(len(unique_types))[inverses]
        node_attr = torch.tensor([[one_hots] * len(X)])[0]

        node_inputs = torch.ones(X.shape)
        labels = torch.ones(len(X))
        labels[: int(len(X) / 2)] = 0.0
        weights = torch.ones(len(X))

        return X, node_attr, labels, weights, node_inputs


class GromacsDatasetFromColvar(BaseAtomisticDataset):
    def __init__(
        self,
        root,
        filenames,
        load_args,
        group,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.filenames = filenames
        self.load_args = load_args
        self.group = group
        super().__init__(root, transform, pre_transform, pre_filter)

    def get_atomistic_data(self):
        from mlcolvar.cvs.committor.utils import compute_committor_weights
        from mlcolvar.utils.io import create_dataset_from_files

        folder = None
        dataset, dataframe = create_dataset_from_files(
            file_names=self.filenames,
            folder=folder,
            create_labels=True,
            # filter_args={
            #     "regex": "phi psi theta ene"
            # },  # to load many positions --> 'regex': 'p[1-9]\.[abc]|p[1-2][0-9]\.[abc]'
            return_dataframe=True,
            load_args=self.load_args,
            verbose=True,
        )

        if "potential.bias" in dataframe:
            dataframe = dataframe.fillna({"potential.bias": 0})  # !!!
        else:
            print("Check me. Bias set to zero")
            dataframe["potential.bias"] = 0

        bias = torch.Tensor(dataframe["potential.bias"].values)

        beta = 1

        dataset = compute_committor_weights(dataset, bias, self.group, beta)

        labels = dataset["labels"]
        weights = dataset["weights"]

        phi, psi, theta, ene = (
            dataframe["phi"],
            dataframe["psi"],
            dataframe["theta"],
            dataframe["ene"],
        )

        # X = torch.zeros((len(phi), 10, 3))
        # for frame in range(len(phi)):
        #     for natom in range(10):
        #         X[frame, natom] = torch.tensor(
        #             [
        #                 dataframe[f"p{natom+1}.a"][frame],
        #                 dataframe[f"p{natom+1}.b"][frame],
        #                 dataframe[f"p{natom+1}.c"][frame],
        #             ]
        #         )
        columns = [
            f"p{natom+1}.{xyz}" for natom in range(10) for xyz in ["a", "b", "c"]
        ]
        X = torch.tensor(dataframe[columns].values).view(len(phi), 10, 3)

        #!!! For position open in ovito the trajectories, I don't think they saved in the colvar... tyer ovito doesnt know the atom types so use the templates to set them; becareful of the strides of when things got saved, time steps of COLVAR might not be same s the trajectories.

        X *= 3.02334
        # Xa = torch.load("X_ala2_A0.npy")
        # node_attra = torch.load("node_attr_ala2_A0.npy")

        # Xb = torch.load("X_ala2_B0.npy")
        # node_attrb = torch.load("node_attr_ala2_B0.npy")

        # Xold = torch.cat([Xa, Xb])

        # node_attr = torch.cat([node_attra, node_attrb])
        # node_attr = node_attr[
        #     :,
        #     :10,
        # ]

        #

        #!!! BECAREFULL ALSO NEED TO INCLDUE TH EOTHER BASSISN !!!!!
        # node_attr = torch.ones((X.shape[0], X.shape[1], 4)) #!!! NEED TO PUT SOME BETTER ONES
        atom_labels = [6, 6, 8, 7, 6, 6, 6, 8, 7, 6]
        unique_types, inverses = np.unique(atom_labels, return_inverse=True)
        one_hots = np.eye(len(unique_types))[inverses]
        node_attr = torch.tensor([[one_hots] * len(X)])[0]

        node_inputs = torch.ones(X.shape)

        return X, node_attr, labels, weights, node_inputs


class OvitoDataset(BaseAtomisticDataset):
    def __init__(
        self,
        root,
        filenames,
        load_args,
        group,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.filenames = filenames
        self.load_args = load_args
        self.group = group
        super().__init__(root, transform, pre_transform, pre_filter)

    def get_atomistic_data(self):

        X_AB = []
        onehot_AB = []

        for i, filename in enumerate(self.filenames):

            # Getting values for only one dump file
            pipeline = import_file(filename)
            pipeline.modifiers.append(UnwrapTrajectoriesModifier()) 
            data = pipeline.compute()

            
            

            timeframes = (
                range(pipeline.source.num_frames)
                if pipeline.source.num_frames > 1
                else range(1)
            )

            X = []
            onehotss=[]
            for timestep in timeframes:
                data = pipeline.compute(timestep)
                xyz = list(data.particles.positions + 0)

                X.append(np.array(xyz))

                atom_labels = torch.tensor(data.particles.particle_types+0)
                unique_types, inverses = torch.unique(atom_labels, return_inverse=True)
                one_hots = torch.eye(len(unique_types))[inverses]
                
                onehotss.append(one_hots.tolist())

            X_AB.append(torch.tensor(X))
            onehot_AB.append(torch.tensor(onehotss))

        X = torch.concat(X_AB)
        node_attr = torch.concat(onehot_AB)


        labels = torch.zeros(len(X))
        labels[len(labels) // 2 :] = 1

        weights = torch.ones(len(X))
        node_inputs = torch.ones(X.shape)


        return X, node_attr, labels, weights, node_inputs


########################################################################
#
########################################################################


class DictCommitorPaperDataset:
    def __init__(
        self,
        root,
        filenames,
        load_args,
        group,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.filenames = filenames
        self.load_args = load_args
        self.group = group

    def get_atomistic_data(self):
        from mlcolvar.cvs.committor.utils import compute_committor_weights
        from mlcolvar.utils.io import create_dataset_from_files

        # temperature
        T = 1
        # Boltzmann factor in the RIGHT ENREGY UNITS!
        kb = 1
        beta = 1 / (kb * T)

        folder = None

        dataset, dataframe = create_dataset_from_files(
            file_names=self.filenames,
            folder=folder,
            create_labels=True,
            filter_args={
                "regex": "p.x|p.y"
            },  # to load many positions --> 'regex': 'p[1-9]\.[abc]|p[1-2][0-9]\.[abc]'
            return_dataframe=True,
            load_args=self.load_args,
            verbose=True,
        )
        # fill empty entries from unbiased simulations

        if "bias" in dataframe:
            dataframe = dataframe.fillna({"bias": 0})
        else:
            print("Check me. Bias set to zero")

            dataframe["bias"] = 0

        bias = torch.Tensor(dataframe["bias"].values)

        dataset = compute_committor_weights(dataset, bias, self.group, beta)

        labels = dataset["labels"]
        weights = dataset["weights"]

        X = dataset["data"]

        # Doesn't work if only one node because edge_vec are zeros

        #!!! ADD BACK FOR 3x3 output
        pos = torch.zeros((X.shape[0], X.shape[1] + 1))
        pos[:, 0] = X[:, 0]
        pos[:, 1] = X[:, 1]
        X = pos
        # X = X.reshape((X.shape[0], 1, X.shape[1]))

        X = torch.zeros((X.shape[0], 2, 3))
        for i in range(len(X)):

            X[i][0] = pos[i]
            X[i][1] = torch.tensor([[1, 1, 1]])
            # torch.cat([pos[i], torch.tensor([[1, 1, 1]])])

        # add = torch.ones((len(X), 1, 3), dtype=torch.get_default_dtype())  #!!!
        # X = torch.concat([X, add], dim=1)
        # X = add.reshape((4000, 2, 3))

        node_attr = torch.tensor(
            [[[1, 0] for _ in range(len(X[0]))] for _ in range(len(X))],
            dtype=torch.get_default_dtype(),
        )
        node_inputs = X

        dataset = DictDataset(
            {
                "pos": X,
                "node_attr": node_attr,
                "labels": labels,
                "weights": weights,
                "node_input": node_inputs,
            }
        )
        return dataset


# class AtomisticDataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.load(self.processed_paths[0])

#     @property
#     def processed_file_names(self):
#         return ["data.pt"]

#     def process(self):
#         # Read data into huge `Data` list.
#         data_list = []

#         nsnapshot = 10
#         X = torch.randn(
#             (nsnapshot, 100, 3), dtype=torch.get_default_dtype()
#         )  # , requires_grad=True

#         labels = torch.zeros((nsnapshot, 1))
#         labels[nsnapshot // 2 :] = 1

#         for n in range(nsnapshot):
#             label = 0 if n < nsnapshot // 2 else 1
#             # weights = compute_committor_weights()

#             graph = Data(
#                 pos=X[n],
#                 node_attr=torch.tensor(
#                     [[1, 0]] * len(X[n]), dtype=torch.get_default_dtype()
#                 ),
#                 labels=labels[n],
#                 # weights = weights
#             )

#             # Add weights
#             graph = compute_committor_weights(
#                 graph, bias=torch.tensor([0]), data_groups=torch.tensor([label]), beta=1
#             )  # !!! make sure to change based on bias beta tec

#             data_list.append(graph)

#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]

#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]

#         self.save(data_list, self.processed_paths[0])


# class CommitorAtomisticDataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.load(self.processed_paths[0])

#     @property
#     def processed_file_names(self):
#         return ["data.pt"]

#     def get_committor_paper_dataset(self):

#         from mlcolvar.data import DictModule
#         from mlcolvar.utils.io import create_dataset_from_files
#         from mlcolvar.cvs.committor.utils import compute_committor_weights

#         # temperature
#         T = 1
#         # Boltzmann factor in the RIGHT ENREGY UNITS!
#         kb = 1
#         beta = 1 / (kb * T)

#         folder = None
#         root_dump = "/home/ubuntu/ksheriff/TRI_PROJECTS/03_plumed/mlcolvar-git/docs/notebooks/tutorials/"
#         filenames = [
#             root_dump + "data/muller-brown/unbiased/state-0/COLVAR",
#             root_dump + "data/muller-brown/unbiased/state-1/COLVAR",
#             # root_dump+'data/muller-brown/biased/committor/iter_1/COLVAR_A',
#             #  root_dump+'data/muller-brown/biased/committor/iter_1/COLVAR_B',
#             #  root_dump+'data/muller-brown/biased/committor/iter_2/COLVAR_A',
#             #  root_dump+'data/muller-brown/biased/committor/iter_2/COLVAR_B'
#         ]

#         load_args = [
#             {"start": 0, "stop": 2000, "stride": 1},
#             {"start": 0, "stop": 2000, "stride": 1},
#             # {'start' : 0, 'stop': 10000, 'stride': 1},
#             #  {'start' : 0, 'stop': 10000, 'stride': 1},
#             #  {'start' : 0, 'stop': 10000, 'stride': 1},
#             #  {'start' : 0, 'stop': 10000, 'stride': 1},
#         ]

#         dataset, dataframe = create_dataset_from_files(
#             file_names=filenames,
#             folder=folder,
#             create_labels=True,
#             filter_args={
#                 "regex": "p.x|p.y"
#             },  # to load many positions --> 'regex': 'p[1-9]\.[abc]|p[1-2][0-9]\.[abc]'
#             return_dataframe=True,
#             load_args=load_args,
#             verbose=True,
#         )
#         # fill empty entries from unbiased simulations
#         # dataframe = dataframe.fillna({'bias': 0}) # !!!
#         dataframe["bias"] = 0

#         bias = torch.Tensor(dataframe["bias"].values)

#         dataset = compute_committor_weights(dataset, bias, [0, 1, 2, 2, 3, 3], beta)

#         # create datamodule with only training set
#         datamodule = DictModule(dataset, lengths=[1])
#         return datamodule

#     def process(self):
#         # Read data into huge `Data` list.
#         data_list = []

#         datamodule = self.get_committor_paper_dataset()
#         X = datamodule.dataset["data"]
#         labels = datamodule.dataset["labels"]
#         weights = datamodule.dataset["weights"]

#         pos = torch.zeros((X.shape[0], X.shape[1] + 1))
#         pos[:, 0] = X[:, 0]
#         pos[:, 1] = X[:, 1]
#         X = pos
#         X = X.reshape((X.shape[0], 1, X.shape[1]))

#         add = torch.randn((len(X), 1, 3), dtype=torch.get_default_dtype())  #!!!
#         add = torch.concat([X, add])
#         X = add.reshape((4000, 2, 3))

#         nsnapshot = len(X)

#         for n in range(nsnapshot):

#             graph = Data(
#                 pos=X[n],
#                 node_attr=torch.tensor(
#                     [[1, 0]] * len(X[n]), dtype=torch.get_default_dtype()
#                 ),
#                 labels=torch.tensor([labels[n]]),
#                 weights=torch.tensor([weights[n]]),
#             )

#             data_list.append(graph)

#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]

#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]

#         self.save(data_list, self.processed_paths[0])
