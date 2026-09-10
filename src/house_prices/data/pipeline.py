from sklearn.model_selection import train_test_split

from house_prices.data.loader import load_train_test
from house_prices.data.preprocessing import get_numeric_categorical_features, build_preprocessor
from house_prices.data.dataset import HousePricesDataset, create_dataloader
from house_prices.data.target import transform_target
from house_prices.data.preprocessing import build_preprocessor, save_preprocessor, load_preprocessor
from house_prices.data.loader import load_test




def prepare_dataloaders(config):
    train, test = load_train_test(config)
    
    X_train = train.drop(columns = [config.data.target, config.data.id_column])
    y_train = train[config.data.target].values
    y_train = transform_target(y_train)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 0.2, random_state= config.general.seed)



    numeric_features, categorical_features = get_numeric_categorical_features(X_train)
        
    processor = build_preprocessor(numeric_features, categorical_features)


    X_train_processed = processor.fit_transform(X_train)
    save_preprocessor(preprocessor=processor, path = config.paths.preprocessor)
    X_val_processed = processor.transform(X_val)
    
    train_dataset = HousePricesDataset(X_train_processed, y_train)
    val_dataset = HousePricesDataset(X_val_processed, y_val)
    
    train_dataloader = create_dataloader(train_dataset, batch_size = config.training.batch_size, shuffle = True)
    val_datloader = create_dataloader(val_dataset, batch_size = config.training.batch_size, shuffle = False)

    return train_dataloader, val_datloader

def prepare_test_dataloader(config):

    test = load_test(config)

    X_test = test.drop(columns=[config.data.id_column])

    processor = load_preprocessor(config.paths.preprocessor)

    X_test_processed = processor.transform(X_test)

    test_dataset = HousePricesDataset(X_test_processed)

    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False)
    return test_dataloader