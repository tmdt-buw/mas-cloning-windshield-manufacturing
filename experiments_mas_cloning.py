import random
import pandas as pd
from sklearn.model_selection import ParameterGrid
from model import MAS
from training import training_loop, create_dataloaders


input_attrs = [
    'sheet_temp_1',
    'sheet_temp_2',
    'sheet_temp_3',
    'chamber_temp_1',
    'chamber_temp_2'
]

output_attrs = ['target']

df_data = pd.read_excel("data.xlsx", index_col="record")
datasets = {}
for name, grouped_data in df_data.groupby(["param_1", "param_2"]):
    X = grouped_data[input_attrs].to_numpy()
    y = grouped_data[output_attrs].to_numpy()
    datasets[str(grouped_data['task_id'].iloc[0])] = {
        'X': X,
        'y': y,
    }


config = {
    'random_state': [100],
    'hidden_dims': [[20, 20]],
    'lr': [0.001],
    'lambda': 1000.,
    'gamma': 1.0,
    'n_inc': 9,
    'n_sequences': 5,
    'use_omega': True,
    'reset_omegas': False,
    'share_reduced_training': 1.0,
    "cloning": True,
    "exp_name": f"mas_cloning"
}

if __name__ == "__main__":
    task_candidates = [
        "t_09",
        "t_04",
        "t_06",
        "t_02",
        "t_01",
        "t_03",
        "t_08",
        "t_05",
        "t_10"
    ]

    list_of_sequences = [random.sample(task_candidates, config['n_inc']) for sequence_number in
                         range(config['n_sequences'])]

    statistics = []

    # Grid search for net parameters
    for hyperparameter in ParameterGrid({'random_state': config['random_state'],
                                         'lr': config['lr'],
                                         'hidden_dims': config['hidden_dims'],
                                         }):
        print(f"Train with parameters: {hyperparameter}")

        for task_sequence_index, task_sequence in enumerate(list_of_sequences):
            model = MAS(input_dim=len(input_attrs),
                        hidden_dims=hyperparameter['hidden_dims'],
                        output_dim=1,
                        heads=[])
            print(f"Train on sequence {task_sequence_index + 1} of {len(list_of_sequences)}: {task_sequence}")

            for task_index, task in enumerate(task_sequence):
                if task == 'None':
                    continue
                share_reduced_training = config['share_reduced_training']
                train_dataloader, val_dataloader, \
                test_dataloader, train_indices, \
                val_indices, test_indices = create_dataloaders(
                    datasets=datasets, task=task,
                    random_state=hyperparameter['random_state'] + task_sequence_index + task_index,
                    share_reduced_training=share_reduced_training)

                model.add_head(head=task, dataloader=val_dataloader, clone=config["cloning"])

                test_loss, \
                val_loss, \
                initial_val_loss, \
                train_losses, \
                train_losses_with_omega, \
                val_losses = training_loop(
                    model=model,
                    lr=hyperparameter['lr'],
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    test_dataloader=test_dataloader,
                    task=task,
                    lamb=config['lambda'],
                    use_omega=config['use_omega'])
                model.update_omega(dataloader=train_dataloader,
                                   task=task,
                                   gamma=config['gamma'],
                                   reset_omegas=config['reset_omegas'])

                model.update_theta()

                single_task_statistics = {
                    'task': task,
                    'task_index': task_index,
                    'task_sequence': task_sequence,
                    'task_sequence_index': task_sequence_index,
                    'random_state': hyperparameter['random_state'] + task_sequence_index + task_index,
                    'model_state_dict': model.state_dict(),
                    'train_indices': train_indices,
                    'test_indices': test_indices,
                    'val_indices': val_indices,
                    'val_loss': val_loss,
                    'test_loss': test_loss,
                    'initial_val_loss': initial_val_loss,
                    'train_losses': train_losses,
                    'train_losses_with_omega': train_losses_with_omega,
                    'val_losses': val_losses,
                    'lr': hyperparameter['lr'],
                    'hidden_dims': hyperparameter['hidden_dims'],
                    'input_attrs': input_attrs,
                    'share_reduced_training': share_reduced_training,

                }
                statistics.append(single_task_statistics)

    exp_name = config["exp_name"]
    data_proportion = config["share_reduced_training"]
    df_results = pd.DataFrame.from_records(statistics)
    df_results.to_excel(f"statistics_{exp_name}.xlsx")
