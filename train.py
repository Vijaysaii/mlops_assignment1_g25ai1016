from sklearn.tree import DecisionTreeRegressor
from misc import load_data, preprocess_data, train_model, evaluate_model


def main():
    print("Loading Boston Housing dataset...")
    df = load_data()

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training Decision Tree Regressor model...")
    model = DecisionTreeRegressor(random_state=42)
    trained_model = train_model(model, X_train, y_train)

    mse = evaluate_model(trained_model, X_test, y_test)
    print(f"Average MSE score on test set (Decision Tree): {mse:.4f}")


if __name__ == "__main__":
    main()
