import splitfolders
# dzielimy dane w proporcjach: 70% train, 15% val, 15% test
splitfolders.ratio("data_deleted", output="data_deleted_split", seed=42, ratio=(.7, .15, .15))